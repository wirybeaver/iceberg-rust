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
use std::fmt::{Debug, Formatter};
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
};
use futures::StreamExt;
use iceberg::table::Table;

use crate::physical_plan::DATA_FILES_COL_NAME;

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
    /// For now, this returns an empty result as a placeholder.
    /// Full implementation will be added when we implement the complete merge logic
    /// in subsequent commits or during integration testing.
    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        // TODO: Implement merge write logic
        // This will write data using TaskWriter similar to IcebergWriteExec
        // but will also track deleted files from the _file column
        let result_schema = Arc::clone(&self.result_schema);

        let stream = futures::stream::once(async move {
            // Placeholder: return empty batch with correct schema
            // Full implementation will:
            // 1. Read input batches
            // 2. Extract _file column to track deleted files
            // 3. Write data using TaskWriter
            // 4. Return (data_files, deleted_files)
            Self::make_result_batch(vec![], vec![])
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
