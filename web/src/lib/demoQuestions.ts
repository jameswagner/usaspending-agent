// Hand-vetted questions - each one verified to trigger the expected agent
// tool call and return a good answer. See backend/app/agent/dev_tools/
// tool_selection_labeled_set.json for the eval set these are drawn from.
export const DEMO_QUESTIONS: string[] = [
  "How is NASA's FY2024 contract spending broken down by NAICS code?",
  "How has NSF's contract spending trended from FY2019 through FY2023?",
  "Which states received the most NSF funding in FY2023?",
  "What is NASA's total budget for FY2024?",
  "How many new grant awards did the Department of Health and Human Services issue in FY2024, and what did they total?",
  "Show whole-of-government spending broken down by budget function for FY2023.",
  "Which federal agency has the biggest budget?",
  "What NAICS code covers custom software development businesses?",
  "What PSC code should I use for office furniture purchases?",
  "What CFDA program number covers Medicaid grants to states?",
  "What FIPS code does USASpending use for Cook County, Illinois?",
  "Show me NASA's five biggest contracts in FY2023.",
  "Tell me the period of performance for NASA's single largest FY2023 contract.",
  "Look up Leidos and tell me its parent company and total transactions all-time.",
  "Show me subawards where Thermo Electron was the subcontractor on Department of Energy prime contracts.",
  "Find NASA's biggest FY2023 contract, then tell me which subcontractors it went to.",
  "What does 'obligation' mean in USASpending data?",
  "What's the difference between a contract and a grant on USASpending?",
  "How many sub-agencies does the Environmental Protection Agency have, according to USASpending?",
  "What percentage of the total federal budget does the Department of Defense account for?",
];

export const DEMO_QUESTIONS_SAMPLE_SIZE = 5;

export function sampleDemoQuestions(count: number = DEMO_QUESTIONS_SAMPLE_SIZE): string[] {
  const shuffled = [...DEMO_QUESTIONS];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled.slice(0, count);
}
