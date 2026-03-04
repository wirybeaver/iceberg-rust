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

//! Test that UPDATE SQL actually works with the TableProvider trait implementation

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::{Int32Array, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::prelude::*;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};
use iceberg_datafusion::IcebergCatalogProvider;
use tempfile::TempDir;

#[tokio::test]
async fn test_update_sql_works() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    let catalog = Arc::new(
        MemoryCatalogBuilder::default()
            .load(
                "memory",
                HashMap::from([(
                    MEMORY_CATALOG_WAREHOUSE.to_string(),
                    temp_dir.path().to_str().unwrap().to_string(),
                )]),
            )
            .await?,
    );

    let namespace = NamespaceIdent::new("test_namespace".to_string());
    catalog.create_namespace(&namespace, HashMap::new()).await?;

    // Create table
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "value", Type::Primitive(PrimitiveType::Int)).into(),
        ])
        .build()?;

    let creation = TableCreation::builder()
        .name("test_table".to_string())
        .schema(schema)
        .location(
            temp_dir
                .path()
                .join("test_table")
                .to_str()
                .unwrap()
                .to_string(),
        )
        .properties(HashMap::new())
        .build();

    catalog.create_table(&namespace, creation).await?;

    // Register catalog with DataFusion
    let iceberg_catalog_provider =
        Arc::new(IcebergCatalogProvider::try_new(catalog.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("iceberg", iceberg_catalog_provider);

    // Insert test data
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, false),
    ]));

    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
        Arc::new(Int32Array::from(vec![1, 2, 3])),
        Arc::new(Int32Array::from(vec![100, 200, 300])),
    ])?;

    let mem_table = Arc::new(datafusion::datasource::MemTable::try_new(
        arrow_schema,
        vec![vec![batch]],
    )?);
    ctx.register_table("source", mem_table)?;

    ctx.sql("INSERT INTO iceberg.test_namespace.test_table SELECT * FROM source")
        .await?
        .collect()
        .await?;

    // Verify initial data
    let results = ctx
        .sql("SELECT * FROM iceberg.test_namespace.test_table ORDER BY id")
        .await?
        .collect()
        .await?;

    assert_eq!(results[0].num_rows(), 3);
    let value_array = results[0]
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(value_array.values(), &[100, 200, 300]);

    // NOW TEST UPDATE SQL!
    println!("Testing UPDATE SQL...");
    let update_result = ctx
        .sql("UPDATE iceberg.test_namespace.test_table SET value = 999 WHERE id = 1")
        .await?
        .collect()
        .await?;

    println!("UPDATE executed successfully! Result: {:?}", update_result);
    println!("Count returned: {:?}", update_result[0].column(0));

    // Give the transaction a moment to complete
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Reload the table to see latest metadata
    let table = catalog
        .load_table(&iceberg::TableIdent::from_strs([
            "test_namespace",
            "test_table",
        ])?)
        .await?;
    println!("Table metadata after UPDATE:");
    println!(
        "  Current snapshot: {:?}",
        table.metadata().current_snapshot().map(|s| s.snapshot_id())
    );

    // Verify updated data
    println!("Querying data after UPDATE...");
    let results = ctx
        .sql("SELECT * FROM iceberg.test_namespace.test_table ORDER BY id")
        .await?
        .collect()
        .await?;

    assert_eq!(results[0].num_rows(), 3);
    let value_array = results[0]
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();

    // Value for id=1 should now be 999!
    assert_eq!(value_array.value(0), 999, "First row should have value=999");
    assert_eq!(value_array.value(1), 200, "Second row should still be 200");
    assert_eq!(value_array.value(2), 300, "Third row should still be 300");

    println!("✅ UPDATE SQL works!");

    Ok(())
}
