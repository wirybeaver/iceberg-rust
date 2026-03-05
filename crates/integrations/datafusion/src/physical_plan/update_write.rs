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
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::str::FromStr;
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{
    DataType, Field, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef,
};
use datafusion::arrow::record_batch::RecordBatchOptions;
use datafusion::common::Result as DFResult;
use datafusion::error::DataFusionError;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
    execute_input_stream,
};
use datafusion::prelude::Expr;
use futures::{StreamExt, TryStreamExt};
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

use crate::physical_plan::expr_to_predicate::convert_filters_to_predicate;
use crate::task_writer::TaskWriter;
use crate::to_datafusion_error;

const DATA_FILES_COL_NAME: &str = "data_files";
const DELETED_FILES_COL_NAME: &str = "deleted_files";

/// An execution plan node that performs UPDATE write operations on an Iceberg table.
///
/// This execution plan implements the write phase of UPDATE using Copy-on-Write (COW) strategy:
/// 1. Executes the filtered input scan to get rows matching the WHERE clause
/// 2. Builds a separate table scan to plan which data files contain matching rows
/// 3. Applies UPDATE column assignments to transform the data
/// 4. Writes modified rows to new data files
/// 5. Tracks original data files for deletion
///
/// The output is a record batch with two columns:
/// - `data_files`: JSON strings representing newly written data files
/// - `deleted_files`: File paths of original data files to be removed
#[derive(Debug)]
pub(crate) struct IcebergUpdateWriteExec {
    table: Table,
    input: Arc<dyn ExecutionPlan>,
    filters: Vec<Expr>,
    assignments: Vec<(String, Arc<dyn PhysicalExpr>)>,
    result_schema: ArrowSchemaRef,
    plan_properties: PlanProperties,
}

impl IcebergUpdateWriteExec {
    pub fn new(
        table: Table,
        input: Arc<dyn ExecutionPlan>,
        filters: Vec<Expr>,
        assignments: Vec<(String, Arc<dyn PhysicalExpr>)>,
        _schema: ArrowSchemaRef, // Unused, kept for API compatibility
    ) -> Self {
        let result_schema = Self::make_result_schema();
        let plan_properties = Self::compute_properties(&input, Arc::clone(&result_schema));

        Self {
            table,
            input,
            filters,
            assignments,
            result_schema,
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

    fn make_result_schema() -> ArrowSchemaRef {
        Arc::new(ArrowSchema::new(vec![
            Field::new(DATA_FILES_COL_NAME, DataType::Utf8, false),
            Field::new(DELETED_FILES_COL_NAME, DataType::Utf8, false),
        ]))
    }

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
                Some("Failed to make UPDATE result batch".to_string()),
            )
        })
    }

    /// Apply UPDATE column assignments to transform the input batch.
    ///
    /// For each row:
    /// - Evaluate the WHERE predicate (if any)
    /// - If the predicate is true (or no predicate), apply assignments
    /// - If the predicate is false, keep the row unchanged
    ///
    /// This ensures non-matching rows are preserved during Copy-on-Write.
    ///
    /// **Performance Optimization**: Returns true if any rows were actually modified,
    /// false if all rows remain unchanged (useful for metrics).
    async fn apply_assignments(
        &self,
        batch: RecordBatch,
        predicate: Option<&Arc<dyn PhysicalExpr>>,
        _context: &Arc<TaskContext>,
    ) -> DFResult<(RecordBatch, bool)> {
        // If there's a predicate, evaluate it to get a boolean mask
        let selection_mask = if let Some(pred) = predicate {
            let result = pred.evaluate(&batch)?;
            let array = result.into_array(batch.num_rows())?;

            // Convert to BooleanArray
            array
                .as_any()
                .downcast_ref::<datafusion::arrow::array::BooleanArray>()
                .ok_or_else(|| {
                    DataFusionError::Internal(
                        "WHERE predicate did not evaluate to boolean".to_string(),
                    )
                })?
                .clone()
        } else {
            // No predicate means all rows match
            datafusion::arrow::array::BooleanArray::from(vec![true; batch.num_rows()])
        };

        // **Early Exit Optimization**: Check if any rows match
        // If no rows match, we can skip assignment evaluation entirely
        let has_matches = selection_mask.iter().any(|v| v == Some(true));

        if !has_matches {
            // No rows match - return unchanged batch
            return Ok((batch, false));
        }

        let mut columns: HashMap<String, ArrayRef> = batch
            .schema()
            .fields()
            .iter()
            .enumerate()
            .map(|(i, field)| (field.name().clone(), batch.column(i).clone()))
            .collect();

        // Apply each assignment, but only to matching rows
        for (column_name, expr) in &self.assignments {
            // Evaluate the assignment expression
            let new_values_result = expr.evaluate(&batch)?;
            let new_values = new_values_result.into_array(batch.num_rows())?;

            // Get the original column
            let original_column = columns.get(column_name).cloned().ok_or_else(|| {
                DataFusionError::Plan(format!("Column {} not found in batch", column_name))
            })?;

            // Use the selection mask to conditionally apply the new values
            // For rows where mask is true, use new_values; otherwise use original_column
            use datafusion::arrow::compute::kernels::zip::zip;
            let updated_column = zip(&selection_mask, &new_values, &original_column)?;

            columns.insert(column_name.clone(), updated_column);
        }

        // Reconstruct RecordBatch with updated columns in original schema order
        let schema = batch.schema();
        let updated_columns: Vec<ArrayRef> = schema
            .fields()
            .iter()
            .map(|field| {
                columns
                    .get(field.name())
                    .cloned()
                    .expect("Column should exist after assignments")
            })
            .collect();

        let result_batch = RecordBatch::try_new_with_options(
            schema,
            updated_columns,
            &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
        )
        .map_err(|e| {
            DataFusionError::ArrowError(
                Box::new(e),
                Some("Failed to apply UPDATE assignments".to_string()),
            )
        })?;

        Ok((result_batch, true))
    }

    /// Get DataFile objects for files that need to be rewritten (for deletion tracking).
    ///
    /// Returns full DataFile objects with proper metadata (size, record count, partition)
    /// needed for RowDelta transaction to mark them as deleted.
    ///
    /// **Performance Optimizations**:
    /// 1. **Partition pruning**: Applies WHERE clause filters to skip entire partitions
    ///    that cannot contain matching rows (e.g., `date = '2024-01-01'` only scans
    ///    that partition).
    ///
    /// 2. **File-level filtering**: Uses manifest statistics (min/max values, null counts)
    ///    to skip files that cannot contain matching rows (e.g., if `WHERE id > 1000`
    ///    and file has `max(id) = 500`, skip the file).
    ///
    /// **Important**: We apply filters for partition/file-level pruning, but NOT for
    /// row-level filtering. In Copy-on-Write mode, we must rewrite entire files to
    /// preserve non-matching rows. Row-level filtering happens in `apply_assignments()`.
    async fn get_deleted_data_files(&self) -> DFResult<Vec<iceberg::spec::DataFile>> {
        use iceberg::spec::{DataFileBuilder, Struct};

        // Convert WHERE clause filters to Iceberg predicates for partition/file pruning
        let predicates = convert_filters_to_predicate(&self.filters);

        // Build table scan with filters for partition pruning and file-level filtering
        // This dramatically reduces files scanned on partitioned tables
        let mut scan_builder = self.table.scan().select_empty();
        if let Some(pred) = predicates {
            scan_builder = scan_builder.with_filter(pred);
        }

        let table_scan = scan_builder.build().map_err(to_datafusion_error)?;

        // Get the planned file scan tasks (already pruned by partitions and statistics)
        let file_scan_tasks = table_scan.plan_files().await.map_err(to_datafusion_error)?;

        let spec_id = self.table.metadata().default_partition_spec_id();

        // Reconstruct DataFile objects from FileScanTask metadata
        let deleted_files: Vec<iceberg::spec::DataFile> = file_scan_tasks
            .map(|task_result| {
                task_result
                    .and_then(|task| {
                        // Reconstruct DataFile with proper metadata for RowDelta transaction
                        DataFileBuilder::default()
                            .file_path(task.data_file_path)
                            .partition_spec_id(spec_id)
                            .partition(task.partition.unwrap_or_else(Struct::empty))
                            .file_format(task.data_file_format)
                            .file_size_in_bytes(task.file_size_in_bytes)
                            .record_count(task.record_count.unwrap_or(0))
                            .content(iceberg::spec::DataContentType::Data)
                            .build()
                            .map_err(|e| {
                                iceberg::Error::new(
                                    iceberg::ErrorKind::Unexpected,
                                    format!("Failed to build DataFile: {e}"),
                                )
                            })
                    })
                    .map_err(to_datafusion_error)
            })
            .try_collect()
            .await?;

        Ok(deleted_files)
    }
}

impl DisplayAs for IcebergUpdateWriteExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => {
                write!(
                    f,
                    "IcebergUpdateWriteExec: table={}, assignments={}",
                    self.table.identifier(),
                    self.assignments.len()
                )
            }
            DisplayFormatType::Verbose => {
                write!(
                    f,
                    "IcebergUpdateWriteExec: table={}, assignments={:?}, result_schema={:?}",
                    self.table.identifier(),
                    self.assignments
                        .iter()
                        .map(|(name, _)| name)
                        .collect::<Vec<_>>(),
                    self.result_schema
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "IcebergUpdateWriteExec: table={}",
                    self.table.identifier()
                )
            }
        }
    }
}

impl ExecutionPlan for IcebergUpdateWriteExec {
    fn name(&self) -> &str {
        "IcebergUpdateWriteExec"
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
                "IcebergUpdateWriteExec expects exactly one child, but provided {}",
                children.len()
            )));
        }

        Ok(Arc::new(Self::new(
            self.table.clone(),
            Arc::clone(&children[0]),
            self.filters.clone(),
            self.assignments.clone(),
            self.schema(),
        )))
    }

    /// Executes the UPDATE write operation for the given partition.
    ///
    /// This function:
    /// 1. Plans which data files contain rows matching the filter
    /// 2. Processes input data from the filtered scan
    /// 3. Applies UPDATE column assignments
    /// 4. Writes modified data to new files using TaskWriter
    /// 5. Returns a batch with new data files and deleted file paths
    ///
    /// Output structure:
    /// ```text
    /// +------------------+------------------+
    /// | data_files       | deleted_files    |
    /// +------------------+------------------+
    /// | "{"file_path":.. | "path/to/old.pq" |
    /// +------------------+------------------+
    /// ```
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
                format!("File format {file_format} is not supported for UPDATE yet!"),
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

        // Get input data stream
        let data = execute_input_stream(
            Arc::clone(&self.input),
            self.input.schema(),
            partition,
            Arc::clone(&context),
        )?;

        // Convert WHERE filters to a single PhysicalExpr predicate
        // We need this to conditionally apply assignments only to matching rows
        let where_predicate: Option<Arc<dyn PhysicalExpr>> = if !self.filters.is_empty() {
            use datafusion::common::DFSchema;
            use datafusion::prelude::SessionContext;

            // Create DFSchema from Arrow schema
            let df_schema = DFSchema::try_from(self.input.schema().as_ref().clone())?;

            // Combine all filters with AND
            let combined_filter = self.filters.iter().cloned().reduce(|acc, filter| {
                datafusion::logical_expr::Expr::BinaryExpr(datafusion::logical_expr::BinaryExpr {
                    left: Box::new(acc),
                    op: datafusion::logical_expr::Operator::And,
                    right: Box::new(filter),
                })
            });

            if let Some(filter_expr) = combined_filter {
                // Convert to physical expression
                let ctx = SessionContext::new();
                let physical_expr = ctx.create_physical_expr(filter_expr, &df_schema)?;
                Some(physical_expr)
            } else {
                None
            }
        } else {
            None
        };

        // Clone needed values for async block
        let self_clone = Arc::new((*self).clone());
        let context_clone = Arc::clone(&context);

        // Create write stream
        let stream = futures::stream::once(async move {
            // Get DataFile objects for files that will be rewritten (with proper metadata)
            let deleted_data_files = self_clone.get_deleted_data_files().await?;

            // Metrics for optimization observability
            let mut total_batches = 0u64;
            let mut batches_with_updates = 0u64;
            let mut total_rows_scanned = 0u64;
            let mut total_rows_updated = 0u64;

            let mut task_writer = task_writer;
            let mut input_stream = data;

            // Process input batches: apply assignments and write
            while let Some(batch) = input_stream.next().await {
                let batch = batch?;
                total_batches += 1;
                total_rows_scanned += batch.num_rows() as u64;

                // Apply UPDATE assignments (conditionally based on WHERE predicate)
                let (updated_batch, has_changes) = self_clone
                    .apply_assignments(batch, where_predicate.as_ref(), &context_clone)
                    .await?;

                if has_changes {
                    batches_with_updates += 1;
                }

                // Count updated rows (approximate - counts all rows in batches with matches)
                if has_changes {
                    total_rows_updated += updated_batch.num_rows() as u64;
                }

                // Write updated data (all rows, both modified and unmodified)
                task_writer
                    .write(updated_batch)
                    .await
                    .map_err(to_datafusion_error)?;
            }

            let data_files = task_writer.close().await.map_err(to_datafusion_error)?;

            // Log optimization metrics
            eprintln!("UPDATE execution metrics:");
            eprintln!("  Files to rewrite: {}", deleted_data_files.len());
            eprintln!("  Total batches processed: {}", total_batches);
            eprintln!(
                "  Batches with updates: {} ({:.1}%)",
                batches_with_updates,
                if total_batches > 0 {
                    (batches_with_updates as f64 / total_batches as f64) * 100.0
                } else {
                    0.0
                }
            );
            eprintln!("  Total rows scanned: {}", total_rows_scanned);
            eprintln!(
                "  Approximate rows updated: {} ({:.1}%)",
                total_rows_updated,
                if total_rows_scanned > 0 {
                    (total_rows_updated as f64 / total_rows_scanned as f64) * 100.0
                } else {
                    0.0
                }
            );

            // Convert new data files to JSON strings
            let data_files_strs: Vec<String> = data_files
                .into_iter()
                .map(|data_file| {
                    serialize_data_file_to_json(data_file, &partition_type, format_version)
                        .map_err(to_datafusion_error)
                })
                .collect::<DFResult<Vec<String>>>()?;

            // Convert deleted data files to JSON strings
            let deleted_files_strs: Vec<String> = deleted_data_files
                .into_iter()
                .map(|data_file| {
                    serialize_data_file_to_json(data_file, &partition_type, format_version)
                        .map_err(to_datafusion_error)
                })
                .collect::<DFResult<Vec<String>>>()?;

            Self::make_result_batch(data_files_strs, deleted_files_strs)
        })
        .boxed();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.result_schema),
            stream,
        )))
    }
}

// Implement Clone manually since PhysicalExpr doesn't implement Clone
impl Clone for IcebergUpdateWriteExec {
    fn clone(&self) -> Self {
        Self {
            table: self.table.clone(),
            input: Arc::clone(&self.input),
            filters: self.filters.clone(),
            assignments: self
                .assignments
                .iter()
                .map(|(name, expr)| (name.clone(), Arc::clone(expr)))
                .collect(),
            result_schema: Arc::clone(&self.result_schema),
            plan_properties: self.plan_properties.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fmt::Formatter;

    use datafusion::arrow::array::Int32Array;
    use datafusion::physical_expr::expressions::Column;
    use datafusion::physical_plan::DisplayAs;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use futures::stream;
    use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
    use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
    use iceberg::{Catalog, CatalogBuilder, MemoryCatalog, NamespaceIdent, TableCreation};
    use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
    use tempfile::TempDir;

    use super::*;

    struct MockExecutionPlan {
        schema: ArrowSchemaRef,
        batches: Vec<RecordBatch>,
        properties: PlanProperties,
    }

    impl MockExecutionPlan {
        fn new(schema: ArrowSchemaRef, batches: Vec<RecordBatch>) -> Self {
            let properties = PlanProperties::new(
                EquivalenceProperties::new(schema.clone()),
                Partitioning::UnknownPartitioning(1),
                EmissionType::Final,
                Boundedness::Bounded,
            );

            Self {
                schema,
                batches,
                properties,
            }
        }
    }

    impl Debug for MockExecutionPlan {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "MockExecutionPlan")
        }
    }

    impl DisplayAs for MockExecutionPlan {
        fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
            write!(f, "MockExecutionPlan")
        }
    }

    impl ExecutionPlan for MockExecutionPlan {
        fn name(&self) -> &str {
            "MockExecutionPlan"
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
            let batches = self.batches.clone();
            let stream = stream::iter(batches.into_iter().map(Ok));
            Ok(Box::pin(RecordBatchStreamAdapter::new(
                self.schema.clone(),
                stream.boxed(),
            )))
        }
    }

    fn temp_path() -> String {
        let temp_dir = TempDir::new().unwrap();
        temp_dir.path().to_str().unwrap().to_string()
    }

    async fn get_iceberg_catalog() -> MemoryCatalog {
        MemoryCatalogBuilder::default()
            .load(
                "memory",
                HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), temp_path())]),
            )
            .await
            .unwrap()
    }

    fn get_test_schema() -> iceberg::Result<Schema> {
        Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "value", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
    }

    fn get_table_creation(
        location: impl ToString,
        name: impl ToString,
        schema: Schema,
    ) -> TableCreation {
        TableCreation::builder()
            .location(location.to_string())
            .name(name.to_string())
            .properties(HashMap::new())
            .schema(schema)
            .build()
    }

    #[tokio::test]
    async fn test_update_write_exec_basic() -> iceberg::Result<()> {
        // 1. Set up test environment
        let iceberg_catalog = get_iceberg_catalog().await;
        let namespace = NamespaceIdent::new("test_namespace".to_string());

        iceberg_catalog
            .create_namespace(&namespace, HashMap::new())
            .await?;

        let schema = get_test_schema()?;
        let table_name = "test_table";
        let table_location = temp_path();
        let creation = get_table_creation(table_location, table_name, schema);
        let table = iceberg_catalog.create_table(&namespace, creation).await?;

        // 2. Create test data
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
            Field::new("value", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "2".to_string(),
            )])),
        ]));

        let id_array = Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef;
        let value_array = Arc::new(Int32Array::from(vec![100, 200, 300])) as ArrayRef;

        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![id_array, value_array])
            .map_err(|e| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("Failed to create batch: {e}"),
                )
            })?;

        // 3. Create mock input plan
        let input_plan = Arc::new(MockExecutionPlan::new(arrow_schema.clone(), vec![batch]));

        // 4. Create UPDATE assignment: SET value = 999
        let value_col_idx = 1;
        let assignments = vec![(
            "value".to_string(),
            Arc::new(Column::new("value", value_col_idx)) as Arc<dyn PhysicalExpr>,
        )];

        // 5. Create UpdateWriteExec
        let update_exec = IcebergUpdateWriteExec::new(
            table,
            input_plan,
            vec![], // no filters for this test
            assignments,
            arrow_schema,
        );

        // 6. Verify schema
        assert_eq!(update_exec.schema().fields().len(), 2);
        assert_eq!(update_exec.schema().field(0).name(), DATA_FILES_COL_NAME);
        assert_eq!(update_exec.schema().field(1).name(), DELETED_FILES_COL_NAME);

        Ok(())
    }
}
