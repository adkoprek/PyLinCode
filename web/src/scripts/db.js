import { schemas } from "./schemas";
import { addRxPlugin, createRxDatabase, promiseWait } from "rxdb";
import { getRxStorageDexie } from "rxdb/plugins/storage-dexie";
import { wrappedValidateAjvStorage } from 'rxdb/plugins/validate-ajv';
import { RxDBDevModePlugin } from 'rxdb/plugins/dev-mode';
import { RxDBQueryBuilderPlugin } from 'rxdb/plugins/query-builder'; 

addRxPlugin(RxDBQueryBuilderPlugin);


let database;
let connected = false;
let connecting = false;

export async function initDatabase() {
    // Handle that db can take a while to connect in different Promises
    if (connecting) {
        await new Promise(resolve => {
            function check() {
                if (connected === true) {
                    resolve();
                } else {
                    requestAnimationFrame(check);
                }
            }
            check();
        });
        return
    }

    if (!connected) {
        connecting = true;
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

        connected = true;
        connecting = false;
    }
}

/******************************* Locks *******************************************/
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

export function subscrbeToInsert(callback) {
    database.locked.insert$.subscribe(change => callback(change.documentId));
}

export function subscrbeToRemove(callback) {
    database.locked.remove$.subscribe(change => callback(change.documentId));
}

/******************************* Submission *******************************************/
export async function addSubmition(submission) {
    await database.submissions.insertIfNotExists(submission)
}

export async function getSubmissions(lesson) {
    return await database.submissions
        .find()
        .where('lessonId')
        .eq(lesson)
        .sort('-timestamp')
        .exec();
}

export function subscribeToSubmissionInsert(callback) {
    database.submissions.insert$.subscribe(change => callback(change.lessonId))
}

/******************************* Current *******************************************/
export async function getCurrentCode(lesson) {
    await database.current.findOne({
        selector: {
            id: lesson
        }
    });
}

export async function ChangeCurrent(lesson) {
    await database.current.upsert(lesson)
}