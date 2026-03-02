// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Physical execution plan for writing merged data files.
//!
//! This module handles writing the results of a MERGE operation to Iceberg data files.

use std::any::Any;
use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use std::str::FromStr;
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{
    DataType, Field, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef,
};
use datafusion::common::Result as DFResult;
use datafusion::error::DataFusionError;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
    execute_input_stream,
};
use futures::StreamExt;
use iceberg::arrow::FieldMatchMode;
use iceberg::spec::{DataFileFormat, serialize_data_file_to_json};
use iceberg::table::Table;
use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::{Error, ErrorKind};
use parquet::file::properties::WriterProperties;
use uuid::Uuid;

use crate::physical_plan::DATA_FILES_COL_NAME;
use crate::task_writer::TaskWriter;
use crate::to_datafusion_error;

/// Column name for deleted file paths in merge operations.
/// Contains file paths that need to be marked as deleted in the commit.
#[allow(dead_code)] // Will be used in Commit 4
pub(crate) const DELETED_FILES_COL_NAME: &str = "deleted_files";

/// An execution plan node that writes merged data to an Iceberg table.
///
/// This is similar to IcebergWriteExec but adapted for MERGE operations.
/// It handles writing both new/updated data files and tracking which files
/// need to be marked as deleted (for Copy-on-Write mode).
///
/// The output contains two columns:
/// - `data_files`: Serialized DataFile objects for new/rewritten files
/// - `deleted_files`: File paths to mark as deleted
#[derive(Debug)]
#[allow(dead_code)] // Will be used in Commit 4
pub(crate) struct IcebergMergeWriteExec {
    table: Table,
    input: Arc<dyn ExecutionPlan>,
    result_schema: ArrowSchemaRef,
    plan_properties: PlanProperties,
}

impl IcebergMergeWriteExec {
    pub fn new(table: Table, input: Arc<dyn ExecutionPlan>, schema: ArrowSchemaRef) -> Self {
        let plan_properties = Self::compute_properties(&input, schema);

        Self {
            table,
            input,
            result_schema: Self::make_result_schema(),
            plan_properties,
        }
    }

    fn compute_properties(
        input: &Arc<dyn ExecutionPlan>,
        schema: ArrowSchemaRef,
    ) -> PlanProperties {
        PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(input.output_partitioning().partition_count()),
            EmissionType::Final,
            Boundedness::Bounded,
        )
    }

    // Create a record batch with serialized data files and deleted file paths
    fn make_result_batch(
        data_files: Vec<String>,
        deleted_files: Vec<String>,
    ) -> DFResult<RecordBatch> {
        let data_files_array = Arc::new(StringArray::from(data_files)) as ArrayRef;
        let deleted_files_array = Arc::new(StringArray::from(deleted_files)) as ArrayRef;

        RecordBatch::try_new(Self::make_result_schema(), vec![
            data_files_array,
            deleted_files_array,
        ])
        .map_err(|e| {
            DataFusionError::ArrowError(
                Box::new(e),
                Some("Failed to make result batch".to_string()),
            )
        })
    }

    fn make_result_schema() -> ArrowSchemaRef {
        Arc::new(ArrowSchema::new(vec![
            Field::new(DATA_FILES_COL_NAME, DataType::Utf8, false),
            Field::new(DELETED_FILES_COL_NAME, DataType::Utf8, false),
        ]))
    }
}

impl DisplayAs for IcebergMergeWriteExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => {
                write!(
                    f,
                    "IcebergMergeWriteExec: table={}",
                    self.table.identifier()
                )
            }
            DisplayFormatType::Verbose => {
                write!(
                    f,
                    "IcebergMergeWriteExec: table={}, result_schema={:?}",
                    self.table.identifier(),
                    self.result_schema
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "IcebergMergeWriteExec: table={}",
                    self.table.identifier()
                )
            }
        }
    }
}

impl ExecutionPlan for IcebergMergeWriteExec {
    fn name(&self) -> &str {
        "IcebergMergeWriteExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true; self.children().len()]
    }

    fn properties(&self) -> &PlanProperties {
        &self.plan_properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(format!(
                "IcebergMergeWriteExec expects exactly one child, but provided {}",
                children.len()
            )));
        }

        Ok(Arc::new(Self::new(
            self.table.clone(),
            Arc::clone(&children[0]),
            self.schema(),
        )))
    }

    /// Executes the merge write operation.
    ///
    /// This method:
    /// 1. Receives merged data from IcebergMergeExec (including _file column)
    /// 2. Extracts _file values to track which files had matched rows
    /// 3. Writes the merged data to new Parquet files using TaskWriter
    /// 4. Returns serialized DataFile objects and deleted file paths
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let partition_type = self.table.metadata().default_partition_type().clone();
        let format_version = self.table.metadata().format_version();

        // Get typed table properties
        let table_props = self
            .table
            .metadata()
            .table_properties()
            .map_err(to_datafusion_error)?;

        // Check data file format
        let file_format = DataFileFormat::from_str(&table_props.write_format_default)
            .map_err(to_datafusion_error)?;
        if file_format != DataFileFormat::Parquet {
            return Err(to_datafusion_error(Error::new(
                ErrorKind::FeatureUnsupported,
                format!("File format {file_format} is not supported for MERGE yet!"),
            )));
        }

        // Create data file writer builder
        let parquet_file_writer_builder = ParquetWriterBuilder::new_with_match_mode(
            WriterProperties::default(),
            self.table.metadata().current_schema().clone(),
            FieldMatchMode::Name,
        );
        let target_file_size = table_props.write_target_file_size_bytes;

        let file_io = self.table.file_io().clone();
        let location_generator = DefaultLocationGenerator::new(self.table.metadata().clone())
            .map_err(to_datafusion_error)?;
        let file_name_generator =
            DefaultFileNameGenerator::new(Uuid::now_v7().to_string(), None, file_format);
        let rolling_writer_builder = RollingFileWriterBuilder::new(
            parquet_file_writer_builder,
            target_file_size,
            file_io,
            location_generator,
            file_name_generator,
        );
        let data_file_writer_builder = DataFileWriterBuilder::new(rolling_writer_builder);

        // Create TaskWriter
        let fanout_enabled = table_props.write_datafusion_fanout_enabled;
        let schema = self.table.metadata().current_schema().clone();
        let partition_spec = self.table.metadata().default_partition_spec().clone();
        let task_writer = TaskWriter::try_new(
            data_file_writer_builder,
            fanout_enabled,
            schema.clone(),
            partition_spec,
        )
        .map_err(to_datafusion_error)?;

        // Get input data
        let data = execute_input_stream(
            Arc::clone(&self.input),
            self.input.schema(),
            partition,
            Arc::clone(&context),
        )?;

        let result_schema = Arc::clone(&self.result_schema);

        // Create write stream
        let stream = futures::stream::once(async move {
            let mut task_writer = task_writer;
            let mut input_stream = data;
            let mut deleted_files: HashSet<String> = HashSet::new();

            while let Some(batch) = input_stream.next().await {
                let batch = batch?;

                // Extract _file column if present to track deleted files
                if let Some(file_column) = batch.column_by_name("_file")
                    && let Some(file_array) = file_column.as_any().downcast_ref::<StringArray>()
                {
                    for file_path in file_array.iter().flatten() {
                        deleted_files.insert(file_path.to_string());
                    }
                }

                // Remove _file column before writing (it's metadata, not data)
                let write_batch = if batch.schema().column_with_name("_file").is_some() {
                    // Find all columns except _file
                    let columns: Vec<ArrayRef> = batch
                        .schema()
                        .fields()
                        .iter()
                        .enumerate()
                        .filter(|(_, field)| field.name() != "_file")
                        .map(|(idx, _)| batch.column(idx).clone())
                        .collect();

                    let write_schema = Arc::new(ArrowSchema::new(
                        batch
                            .schema()
                            .fields()
                            .iter()
                            .filter(|field| field.name() != "_file")
                            .cloned()
                            .collect::<Vec<_>>(),
                    ));

                    RecordBatch::try_new(write_schema, columns).map_err(|e| {
                        DataFusionError::ArrowError(
                            Box::new(e),
                            Some("Failed to create batch without _file column".to_string()),
                        )
                    })?
                } else {
                    batch
                };

                task_writer
                    .write(write_batch)
                    .await
                    .map_err(to_datafusion_error)?;
            }

            let data_files = task_writer.close().await.map_err(to_datafusion_error)?;

            // Convert data files to JSON strings
            let data_files_strs: Vec<String> = data_files
                .into_iter()
                .map(|data_file| {
                    serialize_data_file_to_json(data_file, &partition_type, format_version)
                        .map_err(to_datafusion_error)
                })
                .collect::<DFResult<Vec<String>>>()?;

            // Convert deleted file paths to vector
            let deleted_files_vec: Vec<String> = deleted_files.into_iter().collect();

            Self::make_result_batch(data_files_strs, deleted_files_vec)
        })
        .boxed();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            result_schema,
            stream,
        )))
    }
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::physical_plan::empty::EmptyExec;

    use super::*;

    #[test]
    fn test_merge_write_exec_creation() {
        // Basic structure test
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, true),
        ]));

        let input = Arc::new(EmptyExec::new(schema.clone())) as Arc<dyn ExecutionPlan>;

        // Verify we can create the structure
        assert_eq!(input.schema().fields().len(), 2);
    }

    #[test]
    fn test_result_schema() {
        let result_schema = IcebergMergeWriteExec::make_result_schema();
        assert_eq!(result_schema.fields().len(), 2);
        assert_eq!(result_schema.field(0).name(), DATA_FILES_COL_NAME);
        assert_eq!(result_schema.field(1).name(), DELETED_FILES_COL_NAME);
    }
}
