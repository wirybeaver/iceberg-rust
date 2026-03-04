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

use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, RecordBatch, StringArray, UInt64Array};
use datafusion::arrow::datatypes::{
    DataType, Field, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef,
};
use datafusion::common::{DataFusionError, Result as DFResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::StreamExt;
use iceberg::Catalog;
use iceberg::spec::{DataFile, deserialize_data_file_from_json};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};

use crate::to_datafusion_error;

const DATA_FILES_COL_NAME: &str = "data_files";
const DELETED_FILES_COL_NAME: &str = "deleted_files";

/// IcebergUpdateCommitExec is responsible for committing UPDATE operations
/// using the RowDelta transaction.
///
/// This executor:
/// 1. Collects new data files and deleted DataFile objects from the UpdateWriteExec
/// 2. Creates a RowDelta transaction
/// 3. Adds new data files and removes old data files atomically
/// 4. Returns the count of updated rows
#[derive(Debug)]
pub(crate) struct IcebergUpdateCommitExec {
    table: Table,
    catalog: Arc<dyn Catalog>,
    input: Arc<dyn ExecutionPlan>,
    schema: ArrowSchemaRef,
    count_schema: ArrowSchemaRef,
    plan_properties: PlanProperties,
}

impl IcebergUpdateCommitExec {
    pub fn new(
        table: Table,
        catalog: Arc<dyn Catalog>,
        input: Arc<dyn ExecutionPlan>,
        schema: ArrowSchemaRef,
    ) -> Self {
        let count_schema = Self::make_count_schema();
        let plan_properties = Self::compute_properties(Arc::clone(&count_schema));

        Self {
            table,
            catalog,
            input,
            schema,
            count_schema,
            plan_properties,
        }
    }

    fn compute_properties(schema: ArrowSchemaRef) -> PlanProperties {
        PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        )
    }

    fn make_count_batch(count: u64) -> DFResult<RecordBatch> {
        let count_array = Arc::new(UInt64Array::from(vec![count])) as ArrayRef;

        RecordBatch::try_from_iter_with_nullable(vec![("count", count_array, false)]).map_err(|e| {
            DataFusionError::ArrowError(
                Box::new(e),
                Some("Failed to make UPDATE count batch!".to_string()),
            )
        })
    }

    fn make_count_schema() -> ArrowSchemaRef {
        Arc::new(ArrowSchema::new(vec![Field::new(
            "count",
            DataType::UInt64,
            false,
        )]))
    }
}

impl DisplayAs for IcebergUpdateCommitExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => {
                write!(
                    f,
                    "IcebergUpdateCommitExec: table={}",
                    self.table.identifier()
                )
            }
            DisplayFormatType::Verbose => {
                write!(
                    f,
                    "IcebergUpdateCommitExec: table={}, schema={:?}",
                    self.table.identifier(),
                    self.schema
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "IcebergUpdateCommitExec: table={}",
                    self.table.identifier()
                )
            }
        }
    }
}

impl ExecutionPlan for IcebergUpdateCommitExec {
    fn name(&self) -> &str {
        "IcebergUpdateCommitExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.plan_properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn required_input_distribution(&self) -> Vec<datafusion::physical_plan::Distribution> {
        vec![datafusion::physical_plan::Distribution::SinglePartition; self.children().len()]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(format!(
                "IcebergUpdateCommitExec expects exactly one child, but provided {}",
                children.len()
            )));
        }

        Ok(Arc::new(IcebergUpdateCommitExec::new(
            self.table.clone(),
            self.catalog.clone(),
            children[0].clone(),
            self.schema.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        // IcebergUpdateCommitExec only has one partition (partition 0)
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "IcebergUpdateCommitExec only has one partition, but got partition {partition}"
            )));
        }

        let table = self.table.clone();
        let input_plan = self.input.clone();
        let count_schema = Arc::clone(&self.count_schema);

        let spec_id = self.table.metadata().default_partition_spec_id();
        let partition_type = self.table.metadata().default_partition_type().clone();
        let current_schema = self.table.metadata().current_schema().clone();

        let catalog = Arc::clone(&self.catalog);

        // Process the input stream and commit using RowDelta transaction
        let stream = futures::stream::once(async move {
            let mut new_data_files: Vec<DataFile> = Vec::new();
            let mut deleted_data_files: Vec<DataFile> = Vec::new();
            let mut total_record_count: u64 = 0;

            // Execute and collect results from the input (UpdateWriteExec output)
            let mut batch_stream = input_plan.execute(0, context)?;

            while let Some(batch_result) = batch_stream.next().await {
                let batch = batch_result?;

                // Extract new data files column
                let data_files_array = batch
                    .column_by_name(DATA_FILES_COL_NAME)
                    .ok_or_else(|| {
                        DataFusionError::Internal(
                            "Expected 'data_files' column in UPDATE write batch".to_string(),
                        )
                    })?
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        DataFusionError::Internal(
                            "Expected 'data_files' column to be StringArray".to_string(),
                        )
                    })?;

                // Extract deleted files column
                let deleted_files_array = batch
                    .column_by_name(DELETED_FILES_COL_NAME)
                    .ok_or_else(|| {
                        DataFusionError::Internal(
                            "Expected 'deleted_files' column in UPDATE write batch".to_string(),
                        )
                    })?
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        DataFusionError::Internal(
                            "Expected 'deleted_files' column to be StringArray".to_string(),
                        )
                    })?;

                // Deserialize new data files from JSON
                let batch_new_files: Vec<DataFile> = data_files_array
                    .into_iter()
                    .flatten()
                    .map(|f| -> DFResult<DataFile> {
                        deserialize_data_file_from_json(
                            f,
                            spec_id,
                            &partition_type,
                            &current_schema,
                        )
                        .map_err(to_datafusion_error)
                    })
                    .collect::<DFResult<_>>()?;

                // Deserialize deleted data files from JSON (they're already full DataFile objects)
                let batch_deleted_files: Vec<DataFile> = deleted_files_array
                    .into_iter()
                    .flatten()
                    .map(|f| -> DFResult<DataFile> {
                        deserialize_data_file_from_json(
                            f,
                            spec_id,
                            &partition_type,
                            &current_schema,
                        )
                        .map_err(to_datafusion_error)
                    })
                    .collect::<DFResult<_>>()?;

                // Add record counts from new files to total
                total_record_count += batch_new_files
                    .iter()
                    .map(|f| f.record_count())
                    .sum::<u64>();

                new_data_files.extend(batch_new_files);
                deleted_data_files.extend(batch_deleted_files);
            }

            // If no changes were made, return empty result
            if new_data_files.is_empty() && deleted_data_files.is_empty() {
                return Ok(RecordBatch::new_empty(count_schema));
            }

            // Create a RowDelta transaction
            let tx = Transaction::new(&table);
            let mut row_delta = tx.row_delta();

            // Add new data files
            if !new_data_files.is_empty() {
                row_delta = row_delta.add_data_files(new_data_files);
            }

            // Remove old data files (now with proper metadata!)
            if !deleted_data_files.is_empty() {
                row_delta = row_delta.remove_data_files(deleted_data_files);
            }

            // Apply the action and commit the transaction
            let _updated_table = row_delta
                .apply(tx)
                .map_err(to_datafusion_error)?
                .commit(catalog.as_ref())
                .await
                .map_err(to_datafusion_error)?;

            Self::make_count_batch(total_record_count)
        })
        .boxed();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.count_schema),
            stream,
        )))
    }
}

// Implement Clone manually for ExecutionPlan requirements
impl Clone for IcebergUpdateCommitExec {
    fn clone(&self) -> Self {
        Self {
            table: self.table.clone(),
            catalog: Arc::clone(&self.catalog),
            input: Arc::clone(&self.input),
            schema: Arc::clone(&self.schema),
            count_schema: Arc::clone(&self.count_schema),
            plan_properties: self.plan_properties.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fmt::Formatter;

    use datafusion::arrow::array::ArrayRef;
    use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
    use datafusion::physical_plan::DisplayAs;
    use datafusion::physical_plan::execution_plan::Boundedness;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use futures::stream;
    use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
    use iceberg::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, NestedField, PrimitiveType, Schema,
        Struct, Type,
    };
    use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};

    use super::*;

    struct MockUpdateWriteExec {
        schema: ArrowSchemaRef,
        data_files_json: Vec<String>,
        deleted_files: Vec<String>,
        properties: PlanProperties,
    }

    impl MockUpdateWriteExec {
        fn new(data_files_json: Vec<String>, deleted_files: Vec<String>) -> Self {
            let schema = Arc::new(ArrowSchema::new(vec![
                Field::new(DATA_FILES_COL_NAME, DataType::Utf8, false),
                Field::new(DELETED_FILES_COL_NAME, DataType::Utf8, false),
            ]));

            let properties = PlanProperties::new(
                EquivalenceProperties::new(schema.clone()),
                Partitioning::UnknownPartitioning(1),
                EmissionType::Final,
                Boundedness::Bounded,
            );

            Self {
                schema,
                data_files_json,
                deleted_files,
                properties,
            }
        }
    }

    impl Debug for MockUpdateWriteExec {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "MockUpdateWriteExec")
        }
    }

    impl DisplayAs for MockUpdateWriteExec {
        fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
            write!(f, "MockUpdateWriteExec")
        }
    }

    impl ExecutionPlan for MockUpdateWriteExec {
        fn name(&self) -> &str {
            "MockUpdateWriteExec"
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn properties(&self) -> &PlanProperties {
            &self.properties
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            vec![]
        }

        fn with_new_children(
            self: Arc<Self>,
            _children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> DFResult<Arc<dyn ExecutionPlan>> {
            Ok(self)
        }

        fn execute(
            &self,
            _partition: usize,
            _context: Arc<TaskContext>,
        ) -> DFResult<SendableRecordBatchStream> {
            let data_array = Arc::new(StringArray::from(self.data_files_json.clone())) as ArrayRef;
            let deleted_array = Arc::new(StringArray::from(self.deleted_files.clone())) as ArrayRef;

            let batch = RecordBatch::try_new(self.schema.clone(), vec![data_array, deleted_array])?;

            let stream = stream::once(async move { Ok(batch) }).boxed();
            Ok(Box::pin(RecordBatchStreamAdapter::new(
                self.schema.clone(),
                stream,
            )))
        }
    }

    #[tokio::test]
    async fn test_update_commit_exec() -> iceberg::Result<()> {
        // Create a memory catalog
        let catalog = Arc::new(
            MemoryCatalogBuilder::default()
                .load(
                    "memory",
                    HashMap::from([(
                        MEMORY_CATALOG_WAREHOUSE.to_string(),
                        "memory://root".to_string(),
                    )]),
                )
                .await?,
        );

        // Create namespace and table
        let namespace = NamespaceIdent::new("test_namespace".to_string());
        catalog.create_namespace(&namespace, HashMap::new()).await?;

        let schema = Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "value", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()?;

        let table_creation = TableCreation::builder()
            .name("test_table".to_string())
            .schema(schema)
            .location("memory://root/test_table".to_string())
            .properties(HashMap::new())
            .build();

        let table = catalog.create_table(&namespace, table_creation).await?;

        // Create mock data files (new files from UPDATE write)
        let new_file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("path/to/new_file.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(2048)
            .record_count(150)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::empty())
            .build()
            .unwrap();

        // Serialize to JSON
        let partition_type = table.metadata().default_partition_type().clone();
        let new_file_json = iceberg::spec::serialize_data_file_to_json(
            new_file,
            &partition_type,
            table.metadata().format_version(),
        )?;

        // Create mock deleted data file (with proper metadata, not just path)
        let deleted_file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("path/to/old_file.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(1024)
            .record_count(100)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::empty())
            .build()
            .unwrap();

        let deleted_file_json = iceberg::spec::serialize_data_file_to_json(
            deleted_file,
            &partition_type,
            table.metadata().format_version(),
        )?;

        // Create mock UpdateWriteExec
        let input_exec = Arc::new(MockUpdateWriteExec::new(vec![new_file_json], vec![
            deleted_file_json,
        ]));

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new(DATA_FILES_COL_NAME, DataType::Utf8, false),
            Field::new(DELETED_FILES_COL_NAME, DataType::Utf8, false),
        ]));

        // Create UpdateCommitExec
        let commit_exec =
            IcebergUpdateCommitExec::new(table.clone(), catalog.clone(), input_exec, arrow_schema);

        // Verify schema
        assert_eq!(
            commit_exec.schema(),
            IcebergUpdateCommitExec::make_count_schema()
        );

        // Execute
        let task_ctx = Arc::new(TaskContext::default());
        let stream = commit_exec.execute(0, task_ctx).map_err(|e| {
            iceberg::Error::new(
                iceberg::ErrorKind::Unexpected,
                format!("Failed to execute commit: {e}"),
            )
        })?;
        let batches: Vec<RecordBatch> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<DFResult<_>>()
            .map_err(|e| {
                iceberg::Error::new(
                    iceberg::ErrorKind::Unexpected,
                    format!("Failed to collect batches: {e}"),
                )
            })?;

        // Verify results
        assert_eq!(batches.len(), 1);
        let batch = &batches[0];
        assert_eq!(batch.num_columns(), 1);
        assert_eq!(batch.num_rows(), 1);

        let count_array = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(count_array.value(0), 150); // Record count from new file

        Ok(())
    }
}
