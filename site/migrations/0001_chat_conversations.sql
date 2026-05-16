CREATE TABLE IF NOT EXISTS chat_conversations (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  title TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'web',
  language_mode TEXT,
  island TEXT,
  consent_state TEXT NOT NULL DEFAULT 'sync_training',
  message_count INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  synced_at TEXT,
  metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_chat_conversations_session_updated
  ON chat_conversations(session_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS chat_messages (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  sequence INTEGER NOT NULL,
  client_created_at TEXT,
  model_run TEXT,
  sampler_path TEXT,
  sampler_step TEXT,
  latency_ms INTEGER,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  metadata_json TEXT,
  UNIQUE(conversation_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_sequence
  ON chat_messages(conversation_id, sequence ASC);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
  ON chat_messages(session_id, created_at DESC);

CREATE TABLE IF NOT EXISTS chat_feedback (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  message_id TEXT,
  session_id TEXT NOT NULL,
  rating TEXT NOT NULL,
  correction_text TEXT,
  selected_text TEXT,
  island TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_chat_feedback_conversation
  ON chat_feedback(conversation_id, created_at DESC);

CREATE TABLE IF NOT EXISTS chat_training_examples (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  task_family TEXT NOT NULL DEFAULT 'chat',
  language_mode TEXT,
  messages_json TEXT NOT NULL,
  metadata_json TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_chat_training_examples_session
  ON chat_training_examples(session_id, updated_at DESC);
