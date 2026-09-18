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

// Mirrors backend/app/main.py's GuidedFlowStepRequest/GuidedFlowStepResponse
// and backend/app/agent/guided_flow.py's field names - same manual-sync
// convention as the rest of this file.
export interface GuidedFlowFields {
  state?: string | null;
  // Blank/omitted means "every district in the state" - see
  // guided_flow.py's module docstring.
  district?: string | null;
  start_fiscal_year?: number | null;
  end_fiscal_year?: number | null;
}

export interface GuidedFlowNamedAmount {
  name: string;
  amount: number;
}

export interface GuidedFlowDistrictRow {
  code: string;
  name: string;
  prime_total: number;
  subaward_total: number;
}

export interface GuidedFlowStepRequest {
  conversation_id: string;
  message: string | null;
}

export interface GuidedFlowStepResponse {
  status: "collecting" | "result" | "breakdown" | "aside_answered" | "escaped";
  prompt: string | null;
  fields: GuidedFlowFields;
  prime_total: number | null;
  subaward_total: number | null;
  top_recipients: GuidedFlowNamedAmount[];
  top_subrecipients: GuidedFlowNamedAmount[];
  districts: GuidedFlowDistrictRow[];
  state_prime_total: number | null;
  state_subaward_total: number | null;
  note: string | null;
  tool_citations: ToolCitation[];
  answer: string | null;
  forward_question: string | null;
}
