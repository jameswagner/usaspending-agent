// Mirrors backend/app/agent/response_shaping.py's Pydantic models and
// backend/app/main.py's AskRequest/AskResponse - kept in manual sync,
// there's no shared schema/codegen between the two today.

export interface ChartSpec {
  chart_type: "bar" | "line";
  title: string;
  labels: string[];
  values: number[];
}

export interface Citation {
  chunk_id: string;
  source: string;
  page: number | null;
  term: string | null;
  question: string | null;
  url: string | null;
}

export interface ToolCitation {
  tool_name: string;
  parameters: Record<string, string | number>;
  description: string;
  url: string | null;
  curl: string | null;
}

export interface AskRequest {
  question: string;
  conversation_id: string | null;
}

export interface AskResponse {
  answer_text: string;
  source_type: string;
  conversation_id: string;
  charts: ChartSpec[];
  citations: Citation[];
  tool_citations: ToolCitation[];
}

// The in-flight status of a turn being streamed from POST /ask/stream
// (see web/src/lib/api.ts's askQuestionStream and backend/app/agent/
// streaming.py's SSE event protocol) - "pending" covers the gap before the
// first frame arrives. Irrelevant once response is set (always "done" by
// then), kept mainly for MessageBubble's live status line.
export type TurnStatus =
  | { kind: "pending" }
  | { kind: "tool_call"; toolName: string }
  | { kind: "tool_result"; toolName: string; summary: string }
  | { kind: "tool_error"; toolName: string; message: string }
  | { kind: "done" };

// A UI-level turn - the question plus its response, as rendered in the
// transcript. Not part of the backend API shape. response is null while
// the question is in flight, so it can render immediately. id is a stable
// identity for matching a turn across the streaming status updates that
// replace it in state (useConversation.ts) - the turn object itself gets
// swapped out on every status change, so object reference equality can't
// be used to find "the same turn" again afterward.
export interface ConversationTurn {
  id: string;
  question: string;
  response: AskResponse | null;
  status: TurnStatus;
}
