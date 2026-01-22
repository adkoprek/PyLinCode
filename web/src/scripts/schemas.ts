import { ExtractDocumentTypeFromTypedRxJsonSchema, toTypedRxJsonSchema, RxJsonSchema, RxCollection, RxDatabase, RxDocument } from "rxdb"


const currentSchemaLiteral = {
    title: "Current Code",
    version: 0,
    description: "Stores the current code for each lesson",
    primaryKey: 'id',
    type: 'object',
    properties: {
        id: {
            type: 'string',
            maxLength: 100
        },
        code: {
            type: 'string',
        },
    },
    required: ["id", "code"]
} as const;

const submissionSchemaLiteral = {
    title: "Code Submissions",
    version: 0,
    description: "Stores the successful code submissions for each lesson",
    primaryKey: 'id',
    type: 'object',
    properties: {
        id: {
            type: 'string',
            maxLength: 100
        },
        lessonId: {
            type: 'number',
        },
        code: {
            type: 'string',
        },
        timestamp: {
            type: 'string',
            format: 'date-time'
        }
    },
    required: ["id", "lessonId", "code", "timestamp"]
} as const;

const currentSchemaTyped = toTypedRxJsonSchema(currentSchemaLiteral);
const submissionsSchemaTyped = toTypedRxJsonSchema(submissionSchemaLiteral);

export type CurrentDocType = ExtractDocumentTypeFromTypedRxJsonSchema<typeof currentSchemaTyped>;
export type SubmissionsDocType = ExtractDocumentTypeFromTypedRxJsonSchema<typeof submissionsSchemaTyped>;

export const currentSchema: RxJsonSchema<CurrentDocType> = currentSchemaLiteral;
export const submissionsSchema: RxJsonSchema<SubmissionsDocType> = submissionSchemaLiteral;

export type CurrentDocument = RxDocument<CurrentDocType>;
export type SubmissionsDocument = RxDocument<SubmissionsDocType>;

export type CurrentCollection = RxCollection<CurrentDocType>;
export type SubmissionsCollection = RxCollection<SubmissionsDocType>;

export type DatabaseCollections = {
    current: CurrentCollection,
    submissions: SubmissionsCollection
}

export type Database = RxDatabase<DatabaseCollections>;
