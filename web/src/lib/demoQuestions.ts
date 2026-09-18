// Hand-vetted question templates - every entity below (agency, location,
// category, contractor, subaward pair) was confirmed live against
// api.usaspending.gov to resolve and return nonzero results before being
// added here. See backend/app/agent/dev_tools/tool_selection_labeled_set.json
// for the eval set the original fixed questions were drawn from.

// Counties verified live (FY2023, nonzero spending_by_award_count) for a
// whole-of-government location filter - no agency scoping needed.
const LOCATIONS = [
  "Cook County, Illinois",
  "Harris County, Texas",
  "Los Angeles County, California",
  "Fairfax County, Virginia",
  "Maricopa County, Arizona",
  "Miami-Dade County, Florida",
  "Wayne County, Michigan",
];

// Phrases verified live to resolve to a real NAICS/PSC code (autocomplete is
// a literal substring match against official titles, not semantic) with
// nonzero whole-of-government spending under that code in FY2023.
const CATEGORIES = [
  "office furniture",
  "cloud computing",
  "aircraft parts",
  "environmental remediation",
  "temporary help",
  "research and development in biotechnology",
];

// Recipient names verified live to have a real, unambiguous top match with
// nonzero all-time award amount via /api/v2/recipient/.
const CONTRACTORS = ["Leidos", "Lockheed Martin", "Booz Allen Hamilton", "Northrop Grumman", "General Dynamics"];

// Sub-recipient/prime-agency pairs verified live to return real subaward
// records (spending_level="subawards") - not a cross product of contractors
// and agencies, since most such combinations have no actual subaward data.
interface SubawardPair {
  sub: string;
  primeAgency: string;
}

const SUBAWARD_PAIRS: SubawardPair[] = [
  { sub: "Thermo Electron", primeAgency: "Department of Energy" },
  { sub: "Electric Boat", primeAgency: "Department of Defense" },
  { sub: "Northrop Grumman", primeAgency: "Department of Defense" },
  { sub: "Booz Allen Hamilton", primeAgency: "Department of Defense" },
  { sub: "General Dynamics", primeAgency: "Department of Defense" },
  { sub: "Lockheed Martin", primeAgency: "National Aeronautics and Space Administration" },
];

function pick<T>(items: readonly T[]): T {
  return items[Math.floor(Math.random() * items.length)];
}

interface DemoQuestionTemplate {
  generate: () => string;
}

const TEMPLATES: DemoQuestionTemplate[] = [
  { generate: () => "How is NASA's FY2024 contract spending broken down by NAICS code?" },
  { generate: () => "How has NSF's contract spending trended from FY2019 through FY2023?" },
  { generate: () => "Which states received the most NSF funding in FY2023?" },
  { generate: () => "What is NASA's total budget for FY2024?" },
  {
    generate: () =>
      "How many new grant awards did the Department of Health and Human Services issue in FY2024, and what did they total?",
  },
  { generate: () => "Show whole-of-government spending broken down by budget function for FY2023." },
  { generate: () => "Which federal agency has the biggest budget?" },
  { generate: () => "How much did NSF spend on custom software development contracts in FY2023?" },
  { generate: () => `How much federal spending went toward ${pick(CATEGORIES)} purchases in FY2023?` },
  { generate: () => "Which agencies gave the most Medicaid grants to states in FY2023?" },
  { generate: () => `How much federal spending went to ${pick(LOCATIONS)} in FY2023?` },
  { generate: () => "Show me NASA's five biggest contracts in FY2023." },
  { generate: () => "Tell me the period of performance for the Department of Veterans Affairs' single largest FY2023 contract." },
  { generate: () => `Look up ${pick(CONTRACTORS)} and tell me its parent company and total transactions all-time.` },
  {
    generate: () => {
      const { sub, primeAgency } = pick(SUBAWARD_PAIRS);
      return `Show me subawards where ${sub} was the subcontractor on ${primeAgency} prime contracts.`;
    },
  },
  { generate: () => "Find NASA's biggest FY2023 contract, then tell me which subcontractors it went to." },
  { generate: () => "What's the difference between an obligation and an outlay?" },
  { generate: () => "What's the difference between a contract and a grant?" },
  { generate: () => "How many sub-agencies does the Environmental Protection Agency have?" },
  { generate: () => "What percentage of the total federal budget does the Department of Defense account for?" },
];

export const DEMO_QUESTIONS_SAMPLE_SIZE = 5;

export function sampleDemoQuestions(count: number = DEMO_QUESTIONS_SAMPLE_SIZE): string[] {
  const shuffled = [...TEMPLATES];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled.slice(0, count).map((template) => template.generate());
}
