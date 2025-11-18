export const schemas = [
    {
        locked: {
            schema: {
                title: "Locked Lessons",
                version: 0,
                description: "Stores indices of the locked lessons",
                primaryKey: 'id',
                type: 'object',
                properties: {
                    id: {
                        type: 'string',
                        maxLength: 100
                    },
                },
                required: ["id"]
            }
        }
    },
    {
        current: {
            schema: {
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
            }
        }
    },
    {
        submissions: {
            schema: {
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
            }
        }
    }
]