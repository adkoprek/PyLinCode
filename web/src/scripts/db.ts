import { CurrentDocType, currentSchema, Database, DatabaseCollections, SubmissionsDocType, submissionsSchema } from "./schemas";
import { addRxPlugin, createRxDatabase } from "rxdb";
import { getRxStorageDexie } from "rxdb/plugins/storage-dexie";
import { wrappedValidateAjvStorage } from 'rxdb/plugins/validate-ajv';
import { RxDBDevModePlugin } from 'rxdb/plugins/dev-mode';
import { RxDBQueryBuilderPlugin } from 'rxdb/plugins/query-builder'; 

addRxPlugin(RxDBQueryBuilderPlugin);


let database!:   Database;
let initPromise: Promise<void> | null = null;

export function initDatabase() {
    if (!initPromise) {
        initPromise = init();
    }
    return initPromise;
}

async function init() {
    if (process.env.NODE_ENV === "development") {
        console.log("Initializing database in dev mode...");
        addRxPlugin(RxDBDevModePlugin);
    }

    const storage = wrappedValidateAjvStorage({
        storage: getRxStorageDexie()
    });

    database = await createRxDatabase<DatabaseCollections>({
        name: 'pylinalg',
        storage: storage
    });

    await database.addCollections({
        current: { schema: currentSchema },
        submissions: { schema: submissionsSchema }
    })
}

function db(): Database {
    if (!database) throw new Error("DB not initialized yet");
    return database;
}

export function dbExists() {
    return !!database;
}


/******************************* Submission *******************************************/
export async function addSubmition(submission: SubmissionsDocType) {
    await db().submissions.insertIfNotExists(submission)
}

export function getSubmissions(lesson: number) {
    return db()
        .submissions
        .find()
        .where('lessonId')
        .eq(lesson)
        .sort('-timestamp')
        .exec();
}

export function subscribeToSubmissionInsert(callback: (id: number) => void) {
    database.submissions.insert$.subscribe(change => callback(change.documentData.lessonId));
}

/******************************* Current *******************************************/
export async function getCurrentCode(lesson: number) {
    return db().current.findOne({
        selector: {
            id: lesson.toString()
        }
    }).exec();
}

export async function changeCurrent(current: CurrentDocType) {
    await db().current.upsert(current)
}