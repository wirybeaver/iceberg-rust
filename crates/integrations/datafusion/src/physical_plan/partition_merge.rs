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

//! Partition-aware merge optimization (Storage Partition Join style).
//!
//! This module provides optimization for MERGE operations on partitioned tables
//! by co-locating source and target data by partition, avoiding expensive shuffles.
//!
//! When join keys match partition columns, both source and target can be distributed
//! by partition using hash partitioning, ensuring matching rows are in the same
//! partition without cross-partition communication.

use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::Arc;

use datafusion::error::Result as DFResult;
use datafusion::physical_plan::ExecutionPlan;
use iceberg::spec::{PartitionField, Transform};
use iceberg::table::Table;

use crate::physical_plan::project::project_with_partition;
use crate::physical_plan::repartition::repartition;

/// Optimizes MERGE operation by co-locating source and target data by partition.
///
/// This optimization implements a Storage Partition Join (SPJ) style approach:
/// - When join keys align with partition columns
/// - Adds `_partition` column to both target and source
/// - Repartitions both on `_partition` using hash partitioning
/// - Ensures matching rows are co-located in the same partition
///
/// ## Benefits
///
/// - **Eliminates shuffle**: Matching rows are already in the same partition
/// - **Improves performance**: Reduces network I/O and memory overhead
/// - **Leverages Iceberg partitioning**: Reuses existing partition metadata
///
/// ## When Applied
///
/// Optimization is applied when ALL of the following conditions are met:
/// 1. Target table is partitioned
/// 2. All partition columns are included in the join keys
/// 3. Partition transforms are Identity or Bucket (hash-compatible)
///
/// ## Arguments
///
/// * `target` - Target table execution plan
/// * `source` - Source data execution plan
/// * `join_keys` - Join column pairs as (target_col, source_col)
/// * `table` - Iceberg table with partition specification
/// * `target_partitions` - Number of target partitions for distribution
///
/// ## Returns
///
/// Tuple of (optimized_target, optimized_source) if optimization is applicable,
/// or original plans if optimization cannot be applied.
///
/// ## Example
///
/// ```ignore
/// // For a table partitioned by (region, year):
/// let (target, source) = optimize_merge_for_partitions(
///     target_scan,
///     source_plan,
///     vec![("region", "region"), ("year", "year")],
///     &table,
///     NonZeroUsize::new(8).unwrap(),
/// )?;
/// // Both target and source are now co-located by partition
/// ```
pub(crate) fn optimize_merge_for_partitions(
    target: Arc<dyn ExecutionPlan>,
    source: Arc<dyn ExecutionPlan>,
    join_keys: &[(String, String)],
    table: &Table,
    target_partitions: NonZeroUsize,
) -> DFResult<(Arc<dyn ExecutionPlan>, Arc<dyn ExecutionPlan>)> {
    let metadata = table.metadata();
    let partition_spec = metadata.default_partition_spec();

    // Optimization not applicable for unpartitioned tables
    if partition_spec.is_unpartitioned() {
        return Ok((target, source));
    }

    // Check if optimization is applicable
    if !is_optimization_applicable(partition_spec.fields(), join_keys) {
        return Ok((target, source));
    }

    // Apply optimization: add _partition column and repartition both sides
    let target_with_partition = project_with_partition(target, table)?;
    let target_repartitioned = repartition(
        target_with_partition,
        table.metadata_ref(),
        target_partitions,
    )?;

    // For source, we need to project partition values based on source columns
    // For now, we use the same project_with_partition which expects source schema
    // to match target schema. Full implementation would need custom projection.
    // This is a simplified version that works when source and target schemas match.
    let source_with_partition = project_with_partition(source, table)?;
    let source_repartitioned = repartition(
        source_with_partition,
        table.metadata_ref(),
        target_partitions,
    )?;

    Ok((target_repartitioned, source_repartitioned))
}

/// Checks if partition-aware merge optimization is applicable.
///
/// Returns true if:
/// 1. All partition columns are covered by join keys
/// 2. All partition transforms are hash-compatible (Identity or Bucket)
fn is_optimization_applicable(
    partition_fields: &[PartitionField],
    join_keys: &[(String, String)],
) -> bool {
    // Extract target-side join key column names
    let join_key_names: HashSet<&str> = join_keys
        .iter()
        .map(|(target, _)| target.as_str())
        .collect();

    // Check if all partition fields are covered by join keys
    // and all transforms are hash-compatible
    for field in partition_fields {
        let source_column_name = field.name.as_str();

        // Check if partition column is in join keys
        if !join_key_names.contains(source_column_name) {
            return false;
        }

        // Check if transform is hash-compatible
        match field.transform {
            Transform::Identity | Transform::Bucket(_) => {
                // Hash-compatible transforms
            }
            Transform::Year
            | Transform::Month
            | Transform::Day
            | Transform::Hour
            | Transform::Truncate(_)
            | Transform::Void
            | Transform::Unknown => {
                // Not hash-compatible, skip optimization
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use iceberg::spec::{PartitionField, Transform};

    use super::*;

    #[test]
    fn test_is_optimization_applicable_identity_transform() {
        let partition_fields = vec![PartitionField {
            source_id: 1,
            field_id: 1000,
            name: "region".to_string(),
            transform: Transform::Identity,
        }];

        let join_keys = vec![("region".to_string(), "region".to_string())];

        assert!(is_optimization_applicable(&partition_fields, &join_keys));
    }

    #[test]
    fn test_is_optimization_applicable_bucket_transform() {
        let partition_fields = vec![PartitionField {
            source_id: 1,
            field_id: 1000,
            name: "id".to_string(),
            transform: Transform::Bucket(16),
        }];

        let join_keys = vec![("id".to_string(), "id".to_string())];

        assert!(is_optimization_applicable(&partition_fields, &join_keys));
    }

    #[test]
    fn test_is_optimization_not_applicable_missing_partition_column() {
        let partition_fields = vec![
            PartitionField {
                source_id: 1,
                field_id: 1000,
                name: "region".to_string(),
                transform: Transform::Identity,
            },
            PartitionField {
                source_id: 2,
                field_id: 1001,
                name: "year".to_string(),
                transform: Transform::Year,
            },
        ];

        // Only region in join keys, missing year
        let join_keys = vec![("region".to_string(), "region".to_string())];

        assert!(!is_optimization_applicable(&partition_fields, &join_keys));
    }

    #[test]
    fn test_is_optimization_not_applicable_temporal_transform() {
        let partition_fields = vec![PartitionField {
            source_id: 1,
            field_id: 1000,
            name: "timestamp".to_string(),
            transform: Transform::Day,
        }];

        let join_keys = vec![("timestamp".to_string(), "timestamp".to_string())];

        // Day transform is not hash-compatible
        assert!(!is_optimization_applicable(&partition_fields, &join_keys));
    }

    #[test]
    fn test_is_optimization_applicable_multiple_partitions() {
        let partition_fields = vec![
            PartitionField {
                source_id: 1,
                field_id: 1000,
                name: "region".to_string(),
                transform: Transform::Identity,
            },
            PartitionField {
                source_id: 2,
                field_id: 1001,
                name: "category".to_string(),
                transform: Transform::Bucket(8),
            },
        ];

        let join_keys = vec![
            ("region".to_string(), "region".to_string()),
            ("category".to_string(), "category".to_string()),
        ];

        assert!(is_optimization_applicable(&partition_fields, &join_keys));
    }

    #[test]
    fn test_is_optimization_not_applicable_void_transform() {
        let partition_fields = vec![PartitionField {
            source_id: 1,
            field_id: 1000,
            name: "deleted".to_string(),
            transform: Transform::Void,
        }];

        let join_keys = vec![("deleted".to_string(), "deleted".to_string())];

        assert!(!is_optimization_applicable(&partition_fields, &join_keys));
    }
}
