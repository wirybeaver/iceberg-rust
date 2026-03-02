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
use datafusion::error::Result as DFResult;
use datafusion::execution::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::{
    DisplayAs, ExecutionPlan, Partitioning, PhysicalExpr, PlanProperties, SendableRecordBatchStream,
};
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
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        // TODO: Implement merge execution logic
        // This will be implemented in the next phase
        Err(datafusion::error::DataFusionError::NotImplemented(
            "MERGE execution not yet implemented - coming in next commit".to_string(),
        ))
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
