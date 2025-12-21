PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 30000;
PRAGMA strict = ON;
PRAGMA encoding = 'UTF-8';


CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Experiment keys (for deduplication)
    task_name TEXT NOT NULL,
    model_org TEXT NOT NULL,
    model_ckpt TEXT NOT NULL,
    clf TEXT NOT NULL,
    n_train INTEGER NOT NULL,

    -- Results
    cmap REAL NOT NULL,
    n_classes INTEGER NOT NULL,

    -- Full config as JSON
    exp_cfg TEXT NOT NULL,

    -- Metadata
    argv TEXT NOT NULL,
    git_commit TEXT NOT NULL,
    posix INTEGER NOT NULL,
    gpu_name TEXT,
    hostname TEXT NOT NULL,

    UNIQUE(task_name, model_org, model_ckpt, clf, n_train)
);


CREATE TABLE IF NOT EXISTS predictions (
    example_id TEXT NOT NULL,
    y_true TEXT NOT NULL,
    y_pred TEXT NOT NULL,
    y_score TEXT NOT NULL,

    experiment_id INTEGER NOT NULL,

    PRIMARY KEY (example_id, experiment_id),
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
);


CREATE TABLE IF NOT EXISTS runs (
    task_name TEXT NOT NULL,
    model_org TEXT NOT NULL,
    model_ckpt TEXT NOT NULL,
    clf TEXT NOT NULL,
    n_train INTEGER NOT NULL,

    posix INTEGER NOT NULL,
    pid INTEGER NOT NULL,

    PRIMARY KEY (task_name, model_org, model_ckpt, clf, n_train)
);
