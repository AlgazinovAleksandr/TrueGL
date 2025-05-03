const dbName = "truegl-database";
const collName = "Articles";
const adminDB = db.getSiblingDB(dbName); // connect to the database

// create a table
adminDB.createCollection(collName, {
    validator: {
        $jsonSchema: {
            required: ["link", "source", "status"],
            properties: {
                link: {
                    bsonType: "string"
                },
                source: {
                    bsonType: "string"
                },
                content: {
                    bsonType: "string"
                },
                indices: {
                    bsonType: "array",
                    items: {
                        bsonType: "string"
                    }
                },
                status: {
                    enum: ["processed", "failed to process", "not processed yet"]
                }
            }
        }
    }
});


// create a unique index for the link field (so that we don’t accidentally save the same article URL twice)
adminDB[collName].createIndex(
    { link: 1 },
    { unique: true, name: "unique_link_idx" }
);

print(`Created collection '${dbName}.${collName}' with schema validation and unique index on 'link'.`);
