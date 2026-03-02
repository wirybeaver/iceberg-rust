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

//! Physical execution plan for committing MERGE operations.
//!
//! This module handles committing the results of a MERGE operation using
//! the RowDelta transaction action.

use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, RecordBatch, StringArray, UInt64Array};
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
use iceberg::spec::{DataFile, DataFileBuilder, deserialize_data_file_from_json};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};

use crate::physical_plan::DATA_FILES_COL_NAME;
use crate::physical_plan::merge_write::DELETED_FILES_COL_NAME;
use crate::to_datafusion_error;

/// IcebergMergeCommitExec commits a MERGE operation using RowDelta transaction.
///
/// It collects:
/// - Data files to add (new and rewritten files)
/// - Data files to remove (files containing modified rows in COW mode)
///
/// Then commits them using Transaction::row_delta().
#[derive(Debug)]
#[allow(dead_code)] // Will be used in Commit 4
pub(crate) struct IcebergMergeCommitExec {
    table: Table,
    catalog: Arc<dyn Catalog>,
    input: Arc<dyn ExecutionPlan>,
    schema: ArrowSchemaRef,
    count_schema: ArrowSchemaRef,
    plan_properties: PlanProperties,
}

impl IcebergMergeCommitExec {
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
                Some("Failed to make count batch!".to_string()),
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

impl DisplayAs for IcebergMergeCommitExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => {
                write!(
                    f,
                    "IcebergMergeCommitExec: table={}",
                    self.table.identifier()
                )
            }
            DisplayFormatType::Verbose => {
                write!(
                    f,
                    "IcebergMergeCommitExec: table={}, schema={:?}",
                    self.table.identifier(),
                    self.schema
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "IcebergMergeCommitExec: table={}",
                    self.table.identifier()
                )
            }
        }
    }
}

impl ExecutionPlan for IcebergMergeCommitExec {
    fn name(&self) -> &str {
        "IcebergMergeCommitExec"
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
                "IcebergMergeCommitExec expects exactly one child, but provided {}",
                children.len()
            )));
        }

        Ok(Arc::new(IcebergMergeCommitExec::new(
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
        // IcebergMergeCommitExec only has one partition (partition 0)
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "IcebergMergeCommitExec only has one partition, but got partition {partition}"
            )));
        }

        let table = self.table.clone();
        let input_plan = self.input.clone();
        let count_schema = Arc::clone(&self.count_schema);

        let spec_id = self.table.metadata().default_partition_spec_id();
        let partition_type = self.table.metadata().default_partition_type().clone();
        let current_schema = self.table.metadata().current_schema().clone();

        let catalog = Arc::clone(&self.catalog);

        // Process the input streams and commit using RowDelta
        let stream = futures::stream::once(async move {
            let mut added_data_files: Vec<DataFile> = Vec::new();
            let mut deleted_file_paths: Vec<String> = Vec::new();
            let mut total_record_count: u64 = 0;

            // Execute and collect results from the input
            let mut batch_stream = input_plan.execute(0, context)?;

            while let Some(batch_result) = batch_stream.next().await {
                let batch = batch_result?;

                // Extract data_files column
                let files_array = batch
                    .column_by_name(DATA_FILES_COL_NAME)
                    .ok_or_else(|| {
                        DataFusionError::Internal(
                            "Expected 'data_files' column in input batch".to_string(),
                        )
                    })?
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        DataFusionError::Internal(
                            "Expected 'data_files' column to be StringArray".to_string(),
                        )
                    })?;

                // Deserialize data files
                let batch_files: Vec<DataFile> = files_array
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

                total_record_count += batch_files.iter().map(|f| f.record_count()).sum::<u64>();
                added_data_files.extend(batch_files);

                // Extract deleted_files column
                let deleted_array = batch
                    .column_by_name(DELETED_FILES_COL_NAME)
                    .ok_or_else(|| {
                        DataFusionError::Internal(
                            "Expected 'deleted_files' column in input batch".to_string(),
                        )
                    })?
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        DataFusionError::Internal(
                            "Expected 'deleted_files' column to be StringArray".to_string(),
                        )
                    })?;

                // Collect deleted file paths
                deleted_file_paths
                    .extend(deleted_array.into_iter().flatten().map(|s| s.to_string()));
            }

            // Create DataFile objects for deleted files (needed by RowDelta)
            // Note: We only have file paths, so we create minimal DataFile objects
            // The RowDelta commit will mark these as deleted in the manifest
            let removed_data_files: Vec<DataFile> = deleted_file_paths
                .into_iter()
                .map(|file_path| {
                    // Create a minimal DataFile for deletion tracking
                    // The actual file metadata will be preserved from existing manifests
                    DataFileBuilder::default()
                        .file_path(file_path)
                        .build()
                        .map_err(|e| {
                            DataFusionError::Internal(format!(
                                "Failed to create DataFile for deletion: {e}"
                            ))
                        })
                })
                .collect::<DFResult<_>>()?;

            // If nothing to commit, return empty result
            if added_data_files.is_empty() && removed_data_files.is_empty() {
                return Ok(RecordBatch::new_empty(count_schema));
            }

            // Create a transaction and use row_delta to commit
            let tx = Transaction::new(&table);
            let mut action = tx.row_delta();

            // Add new/rewritten data files
            if !added_data_files.is_empty() {
                action = action.add_data_files(added_data_files);
            }

            // Mark old files as deleted
            if !removed_data_files.is_empty() {
                action = action.remove_data_files(removed_data_files);
            }

            // Apply the action and commit the transaction
            let _updated_table = action
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_schema() {
        let schema = IcebergMergeCommitExec::make_count_schema();
        assert_eq!(schema.fields().len(), 1);
        assert_eq!(schema.field(0).name(), "count");
        assert_eq!(schema.field(0).data_type(), &DataType::UInt64);
    }
}
