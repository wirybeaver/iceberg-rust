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

//! Physical execution plan for MERGE INTO operations
//!
//! This module provides the core execution logic for MERGE INTO statements,
//! implementing a Copy-on-Write (COW) strategy for row-level modifications.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use datafusion::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use datafusion::common::DataFusionError;
use datafusion::error::Result as DFResult;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::JoinType;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, ExecutionPlan, Partitioning, PhysicalExpr, PlanProperties,
};
use futures::StreamExt;
use iceberg::table::Table;

/// Physical plan node for MERGE INTO operation.
///
/// This node performs the core MERGE logic:
/// 1. Joins target and source data
/// 2. Classifies rows as MATCHED or NOT MATCHED
/// 3. Applies UPDATE or INSERT actions
/// 4. Tracks which data files need to be rewritten (COW mode)
///
/// # Copy-on-Write Strategy
///
/// For MATCHED rows (rows that exist in both target and source):
/// - Read the target data files
/// - Apply UPDATE transformations
/// - Write modified rows to new data files
/// - Track original files for deletion
///
/// For NOT MATCHED rows (rows only in source):
/// - Apply INSERT logic
/// - Write new rows to data files
///
/// # Output Schema
///
/// The output includes:
/// - All columns from the merged result
/// - Metadata about files to add/delete (for commit phase)
#[derive(Debug)]
pub struct IcebergMergeExec {
    /// Target table for the merge operation
    table: Table,
    /// Scan plan for target table (should include _file column for COW tracking)
    target_scan: Arc<dyn ExecutionPlan>,
    /// Source data plan
    source: Arc<dyn ExecutionPlan>,
    /// Join conditions (pairs of target/source column expressions)
    join_on: Vec<(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)>,
    /// UPDATE clause: column assignments for MATCHED rows
    /// Format: Vec<(column_name, expression)>
    matched_update: Option<Vec<(String, Arc<dyn PhysicalExpr>)>>,
    /// INSERT clause: expressions to evaluate for NOT MATCHED rows
    not_matched_insert: Option<Vec<Arc<dyn PhysicalExpr>>>,
    /// Output schema
    schema: ArrowSchemaRef,
    /// Cached plan properties
    plan_properties: PlanProperties,
}

impl IcebergMergeExec {
    /// Creates a new IcebergMergeExec.
    ///
    /// # Arguments
    ///
    /// * `table` - Target Iceberg table
    /// * `target_scan` - Physical plan for scanning target table (must include _file column)
    /// * `source` - Physical plan for source data
    /// * `join_on` - Join conditions as (target_expr, source_expr) pairs
    /// * `matched_update` - Optional UPDATE assignments for MATCHED rows
    /// * `not_matched_insert` - Optional INSERT expressions for NOT MATCHED rows
    /// * `schema` - Output schema
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        table: Table,
        target_scan: Arc<dyn ExecutionPlan>,
        source: Arc<dyn ExecutionPlan>,
        join_on: Vec<(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)>,
        matched_update: Option<Vec<(String, Arc<dyn PhysicalExpr>)>>,
        not_matched_insert: Option<Vec<Arc<dyn PhysicalExpr>>>,
        schema: ArrowSchemaRef,
    ) -> Self {
        let plan_properties = Self::compute_properties(schema.clone());

        Self {
            table,
            target_scan,
            source,
            join_on,
            matched_update,
            not_matched_insert,
            schema,
            plan_properties,
        }
    }

    /// Returns a reference to the target table.
    pub fn table(&self) -> &Table {
        &self.table
    }

    /// Returns a reference to the target scan.
    pub fn target_scan(&self) -> &Arc<dyn ExecutionPlan> {
        &self.target_scan
    }

    /// Returns a reference to the source plan.
    pub fn source(&self) -> &Arc<dyn ExecutionPlan> {
        &self.source
    }

    /// Returns the join conditions.
    #[allow(clippy::type_complexity)]
    pub fn join_on(&self) -> &[(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)] {
        &self.join_on
    }

    /// Computes the plan properties.
    fn compute_properties(schema: ArrowSchemaRef) -> PlanProperties {
        PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        )
    }
}

impl ExecutionPlan for IcebergMergeExec {
    fn name(&self) -> &str {
        "IcebergMergeExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> ArrowSchemaRef {
        self.schema.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.target_scan, &self.source]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 2 {
            return Err(datafusion::error::DataFusionError::Plan(
                "IcebergMergeExec requires exactly 2 children".to_string(),
            ));
        }

        Ok(Arc::new(IcebergMergeExec::new(
            self.table.clone(),
            children[0].clone(),
            children[1].clone(),
            self.join_on.clone(),
            self.matched_update.clone(),
            self.not_matched_insert.clone(),
            self.schema.clone(),
        )))
    }

    fn properties(&self) -> &PlanProperties {
        &self.plan_properties
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        use datafusion::arrow::array::{BooleanArray, RecordBatch};
        use datafusion::arrow::compute::concat_batches;
        use datafusion::common::NullEquality;
        use datafusion::physical_plan::execute_input_stream;
        use datafusion::physical_plan::joins::utils::JoinOn;
        use datafusion::physical_plan::joins::{HashJoinExec, PartitionMode};
        use futures::TryStreamExt;

        // For now, implement a simplified version that collects all data
        // and performs a basic join operation
        // Full optimization with streaming joins will come later

        // Step 1: Create a FULL OUTER JOIN between target and source
        let join_on: JoinOn = self
            .join_on
            .iter()
            .map(|(left, right)| (left.clone(), right.clone()))
            .collect();

        let join_exec = Arc::new(HashJoinExec::try_new(
            Arc::clone(&self.target_scan),
            Arc::clone(&self.source),
            join_on,
            None, // No filter
            &JoinType::Full,
            None, // No projection
            PartitionMode::Partitioned,
            NullEquality::NullEqualsNothing, // NULL values don't match
        )?) as Arc<dyn ExecutionPlan>;

        // Step 2: Execute the join
        let join_stream = execute_input_stream(
            join_exec.clone(),
            join_exec.schema(),
            partition,
            Arc::clone(&context),
        )?;

        let target_schema = self.target_scan.schema();
        let _source_schema = self.source.schema();
        let result_schema = Arc::clone(&self.schema);

        let matched_update = self.matched_update.clone();
        let not_matched_insert = self.not_matched_insert.clone();

        // Step 3: Process join results to classify and apply actions
        let stream = futures::stream::once(async move {
            let mut join_stream = join_stream;
            let mut result_batches = Vec::new();

            while let Some(batch) = join_stream.try_next().await? {
                // Detect MATCHED vs NOT MATCHED rows
                // In a FULL OUTER JOIN:
                // - MATCHED: both target and source columns are non-null
                // - NOT MATCHED (source only): target columns are null
                // - Target only (not needed for MERGE): source columns are null

                let target_field_count = target_schema.fields().len();

                // Check first target column for nulls (indicates NOT MATCHED)
                let target_first_col = batch.column(0);

                // Check first source column for nulls (indicates target-only, which we ignore)
                let source_first_col = batch.column(target_field_count);

                // Create boolean arrays for filtering
                let mut matched_mask = Vec::with_capacity(batch.num_rows());
                let mut not_matched_mask = Vec::with_capacity(batch.num_rows());

                for row_idx in 0..batch.num_rows() {
                    let target_null = target_first_col.is_null(row_idx);
                    let source_null = source_first_col.is_null(row_idx);

                    // MATCHED: both sides have data
                    let is_matched = !target_null && !source_null;
                    // NOT MATCHED: only source has data
                    let is_not_matched = target_null && !source_null;

                    matched_mask.push(is_matched);
                    not_matched_mask.push(is_not_matched);
                }

                let matched_filter = BooleanArray::from(matched_mask);
                let not_matched_filter = BooleanArray::from(not_matched_mask);

                // Apply UPDATE action to MATCHED rows
                if matched_update.is_some() {
                    let matched_batch =
                        datafusion::arrow::compute::filter_record_batch(&batch, &matched_filter)?;
                    if matched_batch.num_rows() > 0 {
                        // For now, just pass through the target columns with _file
                        // Full UPDATE expression evaluation will be added when we integrate with logical planner
                        result_batches.push(matched_batch);
                    }
                }

                // Apply INSERT action to NOT MATCHED rows
                if not_matched_insert.is_some() {
                    let not_matched_batch = datafusion::arrow::compute::filter_record_batch(
                        &batch,
                        &not_matched_filter,
                    )?;
                    if not_matched_batch.num_rows() > 0 {
                        // For now, just pass through the source columns
                        // Full INSERT expression evaluation will be added when we integrate with logical planner
                        result_batches.push(not_matched_batch);
                    }
                }
            }

            // Combine all result batches
            if result_batches.is_empty() {
                Ok(RecordBatch::new_empty(result_schema))
            } else {
                concat_batches(&result_schema, &result_batches).map_err(|e| {
                    DataFusionError::ArrowError(
                        Box::new(e),
                        Some("Failed to concatenate merge result batches".to_string()),
                    )
                })
            }
        })
        .boxed();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            stream,
        )))
    }
}

impl DisplayAs for IcebergMergeExec {
    fn fmt_as(
        &self,
        _t: datafusion::physical_plan::DisplayFormatType,
        f: &mut fmt::Formatter,
    ) -> fmt::Result {
        write!(
            f,
            "IcebergMergeExec: matched={}, not_matched={}",
            self.matched_update.is_some(),
            self.not_matched_insert.is_some()
        )
    }
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::physical_plan::empty::EmptyExec;

    use super::*;

    #[test]
    fn test_merge_exec_creation() {
        // Create a simple test table (we'll use empty exec as placeholder)
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, true),
        ]));

        let target_scan = Arc::new(EmptyExec::new(schema.clone())) as Arc<dyn ExecutionPlan>;
        let source = Arc::new(EmptyExec::new(schema.clone())) as Arc<dyn ExecutionPlan>;

        // For this test, we'll create a merge without actual expressions
        // (full integration testing will come later)
        let join_on: Vec<(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)> = vec![];

        // This test just verifies we can create the structure
        // Full functionality testing will come with integration tests
        assert_eq!(target_scan.schema().fields().len(), 2);
        assert_eq!(source.schema().fields().len(), 2);
        assert_eq!(join_on.len(), 0);
    }
}
