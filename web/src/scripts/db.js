import { schemas } from "./schemas";
import { addRxPlugin, createRxDatabase } from "rxdb";
import { getRxStorageDexie } from "rxdb/plugins/storage-dexie";
import { wrappedValidateAjvStorage } from 'rxdb/plugins/validate-ajv';
import { RxDBDevModePlugin } from 'rxdb/plugins/dev-mode';


let database = null;

export async function initDatabase() {
    if (database === null) {
        if (process.env.NODE_ENV === "development") {
            console.log("Initializing database in dev mode...");
            addRxPlugin(RxDBDevModePlugin);
        }
        let storage = getRxStorageDexie();
        storage = wrappedValidateAjvStorage({ storage });
        database = await createRxDatabase({
            name: 'lincode',
            storage: storage
        });

        for (const schema of schemas) {
            await database.addCollections(schema);
        }
    }
}

export async function getLocks() {
    return await database.locked.find().exec()
}

export async function clearLocks() {
    let locks = await getLocks();
    for (const lock of locks) {
        await lock.remove();
    }
}

export async function lockRange(from, to) {
    for (let i = from; i < (to + 1); i++) {
        await database.locked.insertIfNotExists({
            id: i.toString()
        })
    }
}

export async function subscrbeToInsert(callback) {
    database.locked.insert$.subscribe(change => callback(change.documentId));
}

export async function subscrbeToRemove(callback) {
    database.locked.remove$.subscribe(change => callback(change.documentId));
}