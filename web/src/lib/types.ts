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

// A UI-level turn - the question plus its response, as rendered in the
// transcript. Not part of the backend API shape. response is null while
// the question is in flight, so it can render immediately.
export interface ConversationTurn {
  question: string;
  response: AskResponse | null;
}
