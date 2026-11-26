let currentThreadId = localStorage.getItem("travel_thread_id") || null;
let latestAnswerMarkdown = "";
let waitingForApproval = false;

const AGENT_LABELS = {
  flight_agent: "✈️ Flight Agent",
  hotel_agent: "🏨 Hotel Agent",
  weather_agent: "🌦️ Weather Agent",
  budget_agent: "💰 Budget Agent",
  itinerary_agent: "🗓️ Itinerary Agent"
};

function setPrompt(text) {
  document.getElementById("userInput").value = text;
}

function setLoading(isLoading, mode = "draft") {
  const sendBtn = document.getElementById("sendBtn");
  const btnText = document.getElementById("btnText");
  const btnLoader = document.getElementById("btnLoader");
  const approveBtn = document.getElementById("approveBtn");
  const reviseBtn = document.getElementById("reviseBtn");
  const approvalRequest = document.getElementById("approvalRequest");

  sendBtn.disabled = isLoading;
  approveBtn.disabled = isLoading;
  reviseBtn.disabled = isLoading;

  if (isLoading) {
    if (mode === "draft") {
      btnText.classList.add("hidden");
      btnLoader.classList.remove("hidden");
      if (waitingForApproval && approvalRequest) {
        approvalRequest.textContent = "Updating draft itinerary with your latest changes...";
      }
    } else if (mode === "approval") {
      approveBtn.textContent = "Processing...";
      reviseBtn.textContent = "Processing...";
    }
  } else {
    btnText.classList.remove("hidden");
    btnLoader.classList.add("hidden");
    approveBtn.textContent = "Approve & Generate Final";
    reviseBtn.textContent = "Revise Using Feedback";
    if (btnText) {
      btnText.textContent = waitingForApproval ? "Update Plan" : "Generate Draft";
    }
  }
}

function showError(message) {
  const errorBox = document.getElementById("errorBox");
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
  errorBox.scrollIntoView({ behavior: "smooth", block: "center" });
}

function hideError() {
  const errorBox = document.getElementById("errorBox");
  errorBox.classList.add("hidden");
  errorBox.textContent = "";
}

function renderMarkdown(element, markdown) {
  if (typeof marked !== "undefined") {
    element.innerHTML = marked.parse(markdown || "");
  } else {
    element.innerText = markdown || "";
  }
}

function showWorkflow(data) {
  const section = document.getElementById("workflowSection");
  const reasoning = document.getElementById("supervisorReasoning");
  const chips = document.getElementById("agentChips");
  const guardrailBadge = document.getElementById("guardrailBadge");

  reasoning.textContent = data.supervisor_reasoning || "Supervisor routing completed.";
  chips.innerHTML = "";

  (data.selected_agents || []).forEach((agent) => {
    const chip = document.createElement("span");
    chip.className = "agent-chip";
    chip.textContent = AGENT_LABELS[agent] || agent;
    chips.appendChild(chip);
  });

  if (data.guardrail_allowed === false) {
    guardrailBadge.textContent = "Guardrail blocked";
    guardrailBadge.classList.add("blocked");
  } else {
    guardrailBadge.textContent = "Guardrail passed";
    guardrailBadge.classList.remove("blocked");
  }

  section.classList.remove("hidden");
}

function showResult(answer, threadId, isDraft = false, rawData = null) {
  latestAnswerMarkdown = answer || "";

  const resultSection = document.getElementById("resultSection");
  const resultBox = document.getElementById("resultBox");
  const threadInfo = document.getElementById("threadInfo");
  const resultTitle = document.getElementById("resultTitle");

  renderMarkdown(resultBox, latestAnswerMarkdown);
  threadInfo.textContent = `Thread ID: ${threadId}`;
  resultTitle.textContent = isDraft ? "Draft Travel Plan" : "Your Final AI Travel Plan";
  resultSection.classList.remove("hidden");

  // Initialize interactive map visualization
  initTripVisualization(latestAnswerMarkdown, rawData);

  resultSection.scrollIntoView({
    behavior: "smooth",
    block: "start"
  });
}

function showApproval(data) {
  waitingForApproval = true;
  const btnText = document.getElementById("btnText");
  if (btnText) {
    btnText.textContent = "Update Plan";
  }
  const section = document.getElementById("approvalSection");
  const approvalRequest = document.getElementById("approvalRequest");
  approvalRequest.textContent = data.approval_request ||
    "Approve the draft or provide feedback before the final plan is generated.";
  section.classList.remove("hidden");
}

function hideApproval() {
  waitingForApproval = false;
  const btnText = document.getElementById("btnText");
  if (btnText) {
    btnText.textContent = "Generate Draft";
  }
  document.getElementById("approvalSection").classList.add("hidden");
  document.getElementById("approvalFeedback").value = "";
}

async function sendMessage() {
  hideError();

  const sendBtn = document.getElementById("sendBtn");
  if (sendBtn && sendBtn.disabled) {
    return;
  }

  const input = document.getElementById("userInput");
  const message = input.value.trim();

  if (!message) {
    showError("Please enter your travel request first.");
    return;
  }

  setLoading(true, "draft");

  try {
    const response = await fetch("/api/travel", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        message: message,
        thread_id: currentThreadId
      })
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.error || "Something went wrong.");
    }

    currentThreadId = data.thread_id;
    localStorage.setItem("travel_thread_id", currentThreadId);
    input.value = "";

    showWorkflow(data);

    if (data.requires_approval) {
      showResult(data.itinerary || data.answer, data.thread_id, true, data);
      showApproval(data);
    } else {
      hideApproval();
      showResult(data.answer, data.thread_id, false, data);
    }
  } catch (error) {
    showError(error.message);
  } finally {
    setLoading(false, "draft");
  }
}

async function submitApproval(approved) {
  hideError();

  if (!currentThreadId || !waitingForApproval) {
    showError("There is no draft waiting for approval.");
    return;
  }

  const feedbackInput = document.getElementById("approvalFeedback");
  const feedback = feedbackInput.value.trim();

  if (!approved && !feedback) {
    showError("Please enter revision feedback before requesting changes.");
    feedbackInput.focus();
    return;
  }

  setLoading(true, "approval");

  try {
    const response = await fetch("/api/travel/approve", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        thread_id: currentThreadId,
        approved: approved,
        feedback: feedback
      })
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.error || "Could not resume the travel workflow.");
    }

    showWorkflow(data);
    hideApproval();
    showResult(data.answer, data.thread_id, false, data);
  } catch (error) {
    showError(error.message);
  } finally {
    setLoading(false, "approval");
  }
}

function copyResult() {
  const resultBox = document.getElementById("resultBox");
  const text = resultBox.innerText;

  if (!text) {
    return;
  }

  navigator.clipboard.writeText(text)
    .then(() => {
      const copyBtn = document.querySelector(".copy-btn");
      const oldText = copyBtn.textContent;
      copyBtn.textContent = "Copied!";

      setTimeout(() => {
        copyBtn.textContent = oldText;
      }, 1400);
    })
    .catch(() => {
      showError("Could not copy result.");
    });
}

function downloadPDF() {
  const pdfContent = document.getElementById("pdfContent");

  if (!latestAnswerMarkdown || !pdfContent) {
    showError("No travel plan available to download.");
    return;
  }

  const downloadBtn = document.querySelector(".download-btn");
  const oldText = downloadBtn.textContent;
  downloadBtn.textContent = "Preparing PDF...";
  downloadBtn.disabled = true;

  const options = {
    margin: 0.5,
    filename: "ai-travel-plan.pdf",
    image: {
      type: "jpeg",
      quality: 0.98
    },
    html2canvas: {
      scale: 2,
      useCORS: true,
      backgroundColor: "#ffffff"
    },
    jsPDF: {
      unit: "in",
      format: "a4",
      orientation: "portrait"
    },
    pagebreak: {
      mode: ["avoid-all", "css", "legacy"]
    }
  };

  html2pdf()
    .set(options)
    .from(pdfContent)
    .save()
    .then(() => {
      downloadBtn.textContent = oldText;
      downloadBtn.disabled = false;
    })
    .catch(() => {
      downloadBtn.textContent = oldText;
      downloadBtn.disabled = false;
      showError("Could not download PDF.");
    });
}

document.addEventListener("keydown", function(event) {
  if (event.ctrlKey && event.key === "Enter") {
    sendMessage();
  }
});

/* ==========================================================================
   Map-Based Trip Visualization (Leaflet + OpenStreetMap)
   ========================================================================== */

let tripMapInstance = null;
let tripMarkersGroup = null;
let mapMarkersList = [];
let activeDayFilter = "all";
let geocodeQueue = [];
let isGeocodingQueueRunning = false;
let lastGeocodeTimestamp = 0;
const memoryGeoCache = new Map();
const GEO_CACHE_PREFIX = "tripmate_geo_v2_";
const NOMINATIM_MIN_INTERVAL_MS = 1100;

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function normalizeGeoKey(str) {
  return String(str || "")
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "_")
    .replace(/_+/g, "_")
    .slice(0, 80);
}

const GENERIC_LANDMARK_EXCLUSIONS = new Set([
  "city", "neighborhood", "central neighborhood", "local market", "local restaurant",
  "local cafe", "hotel", "hotel option", "hotel suggestions", "hotel recommendations",
  "tbd", "na", "n/a", "airport", "area", "town", "streets", "street", "shops",
  "center", "centre", "surroundings", "district", "downtown", "old town",
  "historic landmarks", "cultural sights", "local dining", "local experiences",
  "top historic landmarks", "morning", "afternoon", "evening", "night", "day",
  "relaxed orientation", "orientation stroll", "souvenir shopping", "relaxing activity",
  "farewell dinner", "traditional dinner", "check in", "freshen up", "departure prep",
  "iconic highlights", "culture", "arrival", "exploration", "free time", "leisure",
  "sightseeing", "flight", "flights", "standard stay", "scenic drive", "coastal drive",
  "beach hopping", "transfer", "back to hotel", "return to hotel", "breakfast at hotel",
  "dinner at hotel", "lunch at hotel", "relax at hotel",
  "activities", "transit", "accommodation", "accommodations", "transport", "transportation",
  "food", "dining", "shopping", "meals", "lunch", "dinner", "breakfast", "snacks",
  "overview", "trip summary", "estimated budget", "budget summary", "final recommendations",
  "practical tips", "notes", "note", "tips"
]);

function normalizeUnicodeSpaces(str) {
  return String(str || "").replace(/[\u00a0\u202f\u2007\u2009\u200a]/g, " ");
}

function isPriceOrNumericNoise(text) {
  if (!text) return true;
  const raw = normalizeUnicodeSpaces(text).trim();
  if (!raw) return true;

  // Reject any currency symbol
  if (/[₹$€£¥฿₩₫₱₺₽₦₴₭₮]/.test(raw)) {
    return true;
  }

  // Reject currency codes and currency words
  if (/\b(?:inr|usd|eur|gbp|jpy|thb|aed|sgd|myr|bdt|idr|vnd|krw|aud|cad|chf|cny|rmb|hkd|nzd|qar|sar|omr|kwd|rupees?|baht|yen|dollars?|euros?|pounds?|dirhams?|taka|lakhs?|crores?)\b/i.test(raw)) {
    return true;
  }
  if (/(?:\brs\.?\s*\d|\d\s*rs\.?\b|^rs\.?$)/i.test(raw)) {
    return true;
  }

  // Reject price/budget/cost/duration phrases
  if (/\b(?:cost|costs|price|prices|pricing|budget|budgets|fare|fares|airfare|fee|fees|ticket|tickets|expense|expenses|subtotal|total|allocation|feasibility|estimate|estimated|approx\.?|per\s+(?:night|person|pax|day|head|couple|room)|hours?|hrs?|mins?|minutes?)\b/i.test(raw)) {
    return true;
  }
  if (/\bday\s*\d+\b|\b\d+\s*days?\b|\b\d+\s*nights?\b/i.test(raw)) {
    return true;
  }

  // Reject numeric/price ranges (e.g., "500-800", "1,500 – 2,500", "10 to 20") and k-shorthand ("50k")
  if (/\b\d[\d,]*(?:\.\d+)?\s*(?:[-–—]|to)\s*\d[\d,]*(?:\.\d+)?\b/i.test(raw)) {
    return true;
  }
  if (/\b\d[\d,]*(?:\.\d+)?\s*k\b/i.test(raw)) {
    return true;
  }

  // Reject strings that are primarily numerical or lack meaningful alphabetic words
  const letters = (raw.match(/[A-Za-z]/g) || []).length;
  const digits = (raw.match(/\d/g) || []).length;
  if (letters < 3) {
    return true;
  }
  if (digits > 0 && digits >= letters) {
    return true;
  }
  if (/^\s*~?\d[\d,.]*\s*[A-Za-z]{0,4}\s*$/.test(raw)) {
    return true;
  }

  return false;
}

function isGenericNoise(text) {
  if (!text) return true;
  const low = normalizeUnicodeSpaces(text).trim().toLowerCase();
  if (low.length < 3) return true;
  if (GENERIC_LANDMARK_EXCLUSIONS.has(low)) return true;
  if (isPriceOrNumericNoise(text)) return true;

  const genericPatterns = [
    "breakfast at hotel", "return to hotel", "back to hotel", "check in at hotel",
    "hotel option", "hotel transfer", "head to hotel", "relax at hotel",
    "reach hotel", "transfer to hotel", "orientation stroll", "local dining",
    "freshen up", "relaxing activity", "farewell dinner", "traditional dinner",
    "souvenir shopping", "beach hopping"
  ];
  return genericPatterns.some((pattern) => low.includes(pattern));
}

function cleanLandmarkName(raw) {
  if (!raw) return null;
  const rawStr = normalizeUnicodeSpaces(raw).trim();
  if (isPriceOrNumericNoise(rawStr)) return null;

  let text = rawStr;
  // Strip lowercase articles from action phrases (e.g., "the Grand Palace"),
  // while preserving capitalized proper names like "The Deck by Arun Residence".
  text = text.replace(/^(?:a|an)\s+/i, "");
  text = text.replace(/^the\s+/, "");
  text = text.replace(/^[*_~`#\s•–—-]+|[*_~`#\s•–—-]+$/g, "");
  text = text.replace(/[.,;:'"()\[\]{}]+$/g, "").trim();

  if (text.length < 3) return null;
  if (isGenericNoise(text)) return null;

  // Must begin with an alphanumeric character
  if (!/^[A-Za-z0-9]/.test(text)) {
    return null;
  }
  return text;
}

function classifyLocation(name, activityContext = "") {
  const nLow = String(name || "").toLowerCase();
  const cLow = String(activityContext || "").toLowerCase();

  // 1. Markets & shopping
  if (["market", "bazaar", "flea", "souk", "arcade", "mall"].some(k => nLow.includes(k))) {
    return "market";
  }

  // 2. Restaurants, cafes, nightlife, clubs
  if (["restaurant", "café", "cafe", "wharf", "diner", "bistro", "bar", "club", "pub", "lounge", "shack", "thalassa", "mambo", "curlies", "britto", "taverna"].some(k => nLow.includes(k))) {
    return "restaurant";
  }

  // 3. Hotels & accommodations
  if (["hotel", "resort", "spa", "inn", "suites", "stay", "villa"].some(k => nLow.includes(k))) {
    return "hotel";
  }

  // 4. Attractions, beaches, forts, landmarks
  if (["beach", "fort", "temple", "church", "cathedral", "falls", "island", "sanctuary", "viewpoint", "museum", "park", "caves", "lighthouse", "point", "cliff"].some(k => nLow.includes(k))) {
    return "attraction";
  }

  // Fallback to activity context clues
  if (["market", "flea", "shopping"].some(k => cLow.includes(k))) {
    return "market";
  }
  if (["dining", "lunch", "dinner", "food", "cuisine", "drinks", "nightlife"].some(k => cLow.includes(k))) {
    return "restaurant";
  }
  if (["hotel", "resort"].some(k => cLow.includes(k))) {
    return "hotel";
  }

  return "attraction";
}

function splitLocationsFromActivity(rawActivity, locationDetails = "") {
  let text = String(rawActivity || "").trim();

  // Strip leading time prefixes (e.g., "09:00 AM - " or "11:30 AM: ")
  text = text.replace(/^(?:\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\s*[-–—:]\s*)/i, "");

  // Normalize delimiters to |SEP|: arrows, plus, ampersands, and 'and' between capitalized words
  text = text.replace(/→/g, " |SEP| ");
  text = text.replace(/\s*(?:->|-->|⇒|&rarr;)\s*/g, " |SEP| ");
  text = text.replace(/\s*\+\s*/g, " |SEP| ");
  text = text.replace(/\s*&\s*/g, " |SEP| ");
  text = text.replace(/([a-z0-9\)])\s+and\s+([A-Z])/g, "$1 |SEP| $2");

  const rawChunks = text.split("|SEP|").map(c => c.trim()).filter(c => c.length > 0);
  const items = [];

  rawChunks.forEach(chunk => {
    let cleanName = chunk;
    let area = "";

    // Check parenthetical area like "Fisherman's Wharf (Calangute)" or "Thalassa (Baga)"
    // Ignore parenthetical price/cost/time notes (e.g., "Wat Pho (Entry ~300 THB)")
    const parenMatch = chunk.match(/^(.*?)\s*\(([^)]+)\)\s*$/);
    if (parenMatch) {
      cleanName = parenMatch[1].trim();
      const parenArea = parenMatch[2].trim();
      if (parenArea && !isGenericNoise(parenArea) && !isPriceOrNumericNoise(parenArea)) {
        area = parenArea;
      }
    } else if (locationDetails && !isGenericNoise(locationDetails) && !isPriceOrNumericNoise(locationDetails)) {
      area = String(locationDetails).trim();
    }

    cleanName = cleanLandmarkName(cleanName);
    if (!cleanName || isGenericNoise(cleanName)) {
      return;
    }

    const category = classifyLocation(cleanName, `${rawActivity} ${locationDetails}`);

    items.push({
      name: cleanName,
      area: area,
      category: category,
      rawActivity: rawActivity,
      locationDetails: locationDetails
    });
  });

  return items;
}

function parseMarkdownTables(text) {
  const lines = String(text || "").split("\n");
  let currentDay = 1;
  const tableRows = [];

  let inTable = false;
  let headerCells = [];
  let actCol = -1;
  let locCol = -1;
  let dayCol = -1;
  let timeCol = -1;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();

    // Check for Day headers outside table
    const dayHeaderMatch = line.match(/#{1,4}\s*Day\s*(\d+)[:\s–—-]*(.*)/i);
    if (dayHeaderMatch) {
      currentDay = parseInt(dayHeaderMatch[1], 10);
      inTable = false;
      continue;
    }

    if (line.startsWith("|") && line.endsWith("|")) {
      const cells = line.split("|").slice(1, -1).map(c => c.trim());
      const nextLine = (i + 1 < lines.length) ? lines[i + 1].trim() : "";
      const isSeparator = /^\|(?:\s*:?-+:?\s*\|)+$/.test(nextLine);

      if (isSeparator) {
        headerCells = cells;
        actCol = -1;
        locCol = -1;
        dayCol = -1;
        timeCol = -1;

        let hasCostHeader = false;
        headerCells.forEach((h, idx) => {
          const hLow = h.toLowerCase();
          if (/activity|attraction|sight|place|plan|schedule|itinerary/i.test(hLow)) {
            actCol = idx;
          } else if (/location|area|destination|venue|address|details|highlight/i.test(hLow)) {
            if (locCol === -1) locCol = idx;
          } else if (/day/i.test(hLow)) {
            dayCol = idx;
          } else if (/time/i.test(hLow)) {
            timeCol = idx;
          } else if (/cost|price|budget|expense|fare|amount|fee|rate/i.test(hLow)) {
            hasCostHeader = true;
          }
        });

        // Skip non-itinerary cost/budget breakdown tables
        if (hasCostHeader && actCol === -1 && dayCol === -1 && timeCol === -1) {
          inTable = false;
          i++; // skip separator line
          continue;
        }

        inTable = true;

        if (actCol === -1) {
          if (headerCells.length >= 3) actCol = (dayCol === 0 || timeCol === 0) ? 1 : 2;
          else if (headerCells.length === 2) actCol = 1;
        }

        i++; // skip separator line
        continue;
      }

      if (/^\|(?:\s*:?-+:?\s*\|)+$/.test(line)) {
        continue;
      }

      if (inTable && cells.length > 0) {
        let rowDay = currentDay;
        if (dayCol >= 0 && cells[dayCol]) {
          const dm = cells[dayCol].match(/Day\s*(\d+)/i);
          if (dm) rowDay = parseInt(dm[1], 10);
        }

        const activityText = (actCol >= 0 && cells[actCol]) ? cells[actCol] : (cells[1] || cells[0]);
        const locationDetails = (locCol >= 0 && cells[locCol]) ? cells[locCol] : "";
        const timeText = (timeCol >= 0 && cells[timeCol]) ? cells[timeCol] : "";

        tableRows.push({
          day: rowDay,
          activity: activityText,
          locationDetails: locationDetails,
          time: timeText
        });
      }
    } else {
      inTable = false;
    }
  }

  return tableRows;
}

function normalizeDestination(rawDest) {
  let dest = String(rawDest || "").trim();
  if (!dest) return "Selected Destination";

  dest = dest.replace(/\([^)]*\)/g, "").trim();

  // If destination mentions Goa without country, append India
  if (/goa/i.test(dest) && !/india/i.test(dest)) {
    return `${dest}, Goa, India`;
  }
  // If destination mentions a city without country
  if (/tokyo|osaka|kyoto/i.test(dest) && !/japan/i.test(dest)) {
    return `${dest}, Japan`;
  }
  if (/paris/i.test(dest) && !/france/i.test(dest)) {
    return `${dest}, France`;
  }
  if (/dubai/i.test(dest) && !/uae|emirates/i.test(dest)) {
    return `${dest}, UAE`;
  }
  if (/bangkok|phuket|chiang mai|krabi|pattaya|koh samui/i.test(dest) && !/thailand/i.test(dest)) {
    return `${dest}, Thailand`;
  }

  return dest;
}

function parseItineraryLocations(markdownText, data) {
  const text = markdownText || "";
  const locations = [];
  const unmapped = [];
  const seenKeys = new Set();
  const daysFound = new Set();

  // 1. Resolve destination
  let destination = "";
  if (data && data.trip_constraints && data.trip_constraints.destination) {
    destination = data.trip_constraints.destination;
  } else if (data && data.structured_trip_state && data.structured_trip_state.destination) {
    const d = data.structured_trip_state.destination;
    destination = d.city || d.country || d.raw_text || "";
  }
  if (!destination) {
    const titleMatch = text.match(/#\s*(?:Comprehensive Travel Plan|Draft Itinerary):\s*([^\n\r]+)/i);
    if (titleMatch) {
      destination = titleMatch[1];
    }
  }
  destination = normalizeDestination(destination);

  // 2. Extract hotels from suggestions (strictly reject generic "Hotel Option")
  const hotelSources = [];
  if (data && data.hotel_results) {
    hotelSources.push(data.hotel_results);
  }
  const hotelSectionMatch = text.match(/##\s*3\.\s*Hotel Suggestions([\s\S]*?)(?=##\s*\d|$)/i);
  if (hotelSectionMatch) {
    hotelSources.push(hotelSectionMatch[1]);
  }

  const genericHotelPlaceholders = new Set([
    "hotel option", "hotel", "hotel suggestions", "hotel recommendations",
    "recommended hotel", "budget hotel", "luxury hotel", "hotel name",
    "hotel options", "tbd", "na", "n/a"
  ]);

  hotelSources.forEach((source) => {
    const lines = String(source).split("\n");
    lines.forEach((line) => {
      const match = line.match(/[-*]\s*\*\*([^*]+)\*\*[:\s–—-]*(.*)/i);
      if (match) {
        const candidate = cleanLandmarkName(match[1]);
        if (candidate && !genericHotelPlaceholders.has(candidate.toLowerCase()) && !isGenericNoise(candidate)) {
          const key = candidate.toLowerCase();
          if (!seenKeys.has(key)) {
            seenKeys.add(key);
            locations.push({
              id: `hotel_${locations.length}`,
              name: candidate,
              area: "",
              category: "hotel",
              day: null,
              snippet: match[2] ? match[2].slice(0, 140) : "Recommended accommodation"
            });
          }
        }
      }
    });
  });

  // 3. Extract from Markdown Tables
  const tableRows = parseMarkdownTables(text);
  if (tableRows.length > 0) {
    tableRows.forEach((row) => {
      daysFound.add(row.day);
      const splitItems = splitLocationsFromActivity(row.activity, row.locationDetails);

      splitItems.forEach((item) => {
        const key = item.name.toLowerCase();
        if (!seenKeys.has(key)) {
          seenKeys.add(key);
          const snippetText = [row.time, item.rawActivity, row.locationDetails]
            .filter(Boolean)
            .join(" • ")
            .slice(0, 140);

          locations.push({
            id: `loc_${row.day}_${locations.length}`,
            name: item.name,
            area: item.area,
            category: item.category,
            day: row.day,
            snippet: snippetText
          });
        }
      });
    });
  }

  // 4. Fallback / supplementary: Parse Day Headings and bullet points if no tables or for extra days
  const dayRegex = /(?:^|\n)#{1,4}\s*Day\s*(\d+)[:\s–—-]*(.*?)(?=(?:\n#{1,4}\s*Day\s*\d+|\n##\s|$))/gis;
  let dayMatch;

  while ((dayMatch = dayRegex.exec(text)) !== null) {
    const dayNumber = parseInt(dayMatch[1], 10);
    daysFound.add(dayNumber);
    const dayContent = dayMatch[2] || "";
    const lines = dayContent.split("\n");

    lines.forEach((line) => {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#") || trimmed.startsWith("|")) return;

      // Check bold items in bullet lines
      const boldMatches = trimmed.matchAll(/\*\*([^*]+)\*\*/g);
      for (const bm of boldMatches) {
        const candidate = cleanLandmarkName(bm[1]);
        if (candidate && !genericHotelPlaceholders.has(candidate.toLowerCase()) && !isGenericNoise(candidate)) {
          const key = candidate.toLowerCase();
          if (!seenKeys.has(key)) {
            seenKeys.add(key);
            locations.push({
              id: `attr_${dayNumber}_${locations.length}`,
              name: candidate,
              area: "",
              category: classifyLocation(candidate, trimmed),
              day: dayNumber,
              snippet: trimmed.replace(/[-*#]/g, "").slice(0, 140)
            });
          }
        }
      }

      // Check action verbs
      const actionMatches = trimmed.matchAll(
        /(?:visit|explore|discover|tour|head to|walk (?:through|around)|see|stroll (?:around|through)|cross|view from|ascend|stop at|check out)\s+([A-Z0-9][A-Za-z0-9\s'’&@.-]{2,45}?)(?=(?:,\s*(?:and\s+)?|\s+and\s+|\s+for\s+|\s+to\s+|\s+in\s+|\s+from\s+|\.|\n|\(|$))/gi
      );
      for (const am of actionMatches) {
        const candidate = cleanLandmarkName(am[1]);
        if (candidate && !genericHotelPlaceholders.has(candidate.toLowerCase()) && !isGenericNoise(candidate)) {
          const key = candidate.toLowerCase();
          if (!seenKeys.has(key)) {
            seenKeys.add(key);
            locations.push({
              id: `attr_${dayNumber}_${locations.length}`,
              name: candidate,
              area: "",
              category: classifyLocation(candidate, trimmed),
              day: dayNumber,
              snippet: trimmed.replace(/[-*#]/g, "").slice(0, 140)
            });
          }
        }
      }
    });
  }

  return {
    destination: destination,
    locations: locations,
    unmapped: unmapped,
    days: Array.from(daysFound).sort((a, b) => a - b)
  };
}

function getCachedGeocode(cacheKey) {
  if (memoryGeoCache.has(cacheKey)) {
    return memoryGeoCache.get(cacheKey);
  }
  try {
    const raw = localStorage.getItem(cacheKey);
    if (raw) {
      const parsed = JSON.parse(raw);
      memoryGeoCache.set(cacheKey, parsed);
      return parsed;
    }
  } catch (e) {
    // localStorage may be disabled or full
  }
  return null;
}

function setCachedGeocode(cacheKey, data) {
  memoryGeoCache.set(cacheKey, data);
  try {
    localStorage.setItem(cacheKey, JSON.stringify(data));
  } catch (e) {
    // localStorage may be disabled or full
  }
}

function geocodeLocation(item, destination) {
  const cacheKey = `${GEO_CACHE_PREFIX}${normalizeGeoKey(item.name)}__${normalizeGeoKey(item.area)}__${normalizeGeoKey(destination)}`;
  const cached = getCachedGeocode(cacheKey);
  if (cached) {
    return Promise.resolve(cached);
  }

  return new Promise((resolve) => {
    geocodeQueue.push({
      item,
      destination,
      cacheKey,
      resolve
    });
    triggerGeocodeQueue();
  });
}

function buildGeocodeQueries(item, dest) {
  const cleanDest = (dest && dest !== "Selected Destination") ? dest.trim() : "";
  const hasArea = Boolean(item.area && item.area !== cleanDest && !isPriceOrNumericNoise(item.area));
  const queries = [];
  const seen = new Set();

  const addQuery = (namePart, areaPart = "") => {
    const n = String(namePart || "").trim();
    if (!n || n.length < 3 || isPriceOrNumericNoise(n)) return;
    const parts = [n];
    if (areaPart && areaPart.toLowerCase() !== cleanDest.toLowerCase()) {
      parts.push(areaPart);
    }
    if (cleanDest && !n.toLowerCase().includes(cleanDest.toLowerCase())) {
      parts.push(cleanDest);
    }
    const q = parts.join(", ");
    const key = q.toLowerCase();
    if (!seen.has(key)) {
      seen.add(key);
      queries.push(q);
    }
  };

  // Tier 1: Name + Area + Destination (or Name + Destination)
  if (hasArea) {
    addQuery(item.name, item.area);
  }
  // Tier 2: Name + Destination
  addQuery(item.name);

  // Tier 3: Normalized variants while preserving destination context
  const baseName = String(item.name || "").trim();

  // 3a. Strip leading descriptor prefixes (e.g., "Restaurant ...", "Hotel ...")
  const strippedPrefix = baseName.replace(/^(?:café|cafe|club|restaurant|hotel|resort|bar|the)\s+/i, "").trim();
  if (strippedPrefix.length >= 3 && strippedPrefix.toLowerCase() !== baseName.toLowerCase()) {
    if (hasArea) addQuery(strippedPrefix, item.area);
    addQuery(strippedPrefix);
  }

  // 3b. Try prepending "The " when omitted (e.g., "Deck by Arun Residence" -> "The Deck by Arun Residence")
  if (!/^the\s+/i.test(baseName)) {
    addQuery(`The ${baseName}`);
  }

  // 3c. Handle "@..." venue suffixes (e.g., "Kan Eang@Pier" -> "Kan Eang", "Kan Eang Pier")
  if (baseName.includes("@")) {
    const withoutAtSuffix = baseName.replace(/\s*@.*$/, "").trim();
    const spaceSeparated = baseName.replace(/\s*@\s*/g, " ").trim();
    addQuery(withoutAtSuffix);
    addQuery(spaceSeparated);
  }

  // 3d. Handle "<Venue> by <Parent>" names (e.g., "Deck by Arun Residence" -> "Arun Residence", "The Deck")
  const byMatch = baseName.match(/^(.+?)\s+by\s+(.+)$/i);
  if (byMatch) {
    const parentVenue = byMatch[2].trim();
    const childVenue = byMatch[1].trim();
    addQuery(parentVenue);
    addQuery(childVenue);
    if (!/^the\s+/i.test(childVenue)) {
      addQuery(`The ${childVenue}`);
    }
  }

  // 3e. Strip trailing English generic type words when local name alone is indexed (e.g., "Suan Rot Fai Park" -> "Suan Rot Fai")
  const strippedSuffix = baseName.replace(/\s+(?:park|market|garden|gardens|pier)$/i, "").trim();
  if (strippedSuffix.length >= 4 && strippedSuffix.toLowerCase() !== baseName.toLowerCase()) {
    addQuery(strippedSuffix);
  }

  return queries;
}

async function fetchNominatimQuery(query) {
  const url = `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(query)}&format=json&limit=1`;
  const attempt = async () => {
    const response = await fetch(url, {
      headers: { "Accept": "application/json" }
    });
    if (response.status === 429 || response.status === 503) {
      return { retry: true, status: response.status };
    }
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return { retry: false, data: await response.json() };
  };

  let res = await attempt();
  if (res.retry) {
    await new Promise((r) => setTimeout(r, 2000));
    lastGeocodeTimestamp = Date.now();
    res = await attempt();
    if (res.retry) {
      throw new Error(`HTTP ${res.status}`);
    }
  }
  return res.data;
}

async function triggerGeocodeQueue() {
  if (isGeocodingQueueRunning) return;
  isGeocodingQueueRunning = true;

  while (geocodeQueue.length > 0) {
    const task = geocodeQueue.shift();
    const item = task.item;
    const dest = task.destination;

    try {
      const queries = buildGeocodeQueries(item, dest);
      let results = [];

      for (let idx = 0; idx < queries.length; idx++) {
        const now = Date.now();
        const elapsed = now - lastGeocodeTimestamp;
        if (elapsed < NOMINATIM_MIN_INTERVAL_MS) {
          await new Promise((r) => setTimeout(r, NOMINATIM_MIN_INTERVAL_MS - elapsed));
        }
        lastGeocodeTimestamp = Date.now();
        results = await fetchNominatimQuery(queries[idx]);
        if (Array.isArray(results) && results.length > 0) {
          break;
        }
      }

      if (Array.isArray(results) && results.length > 0) {
        const topResult = results[0];
        const resData = {
          found: true,
          lat: parseFloat(topResult.lat),
          lon: parseFloat(topResult.lon),
          displayName: topResult.display_name
        };
        setCachedGeocode(task.cacheKey, resData);
        task.resolve(resData);
      } else {
        const notFoundData = {
          found: false,
          reason: "Coordinates not found on OpenStreetMap"
        };
        setCachedGeocode(task.cacheKey, notFoundData);
        task.resolve(notFoundData);
      }
    } catch (err) {
      const errData = {
        found: false,
        reason: `Geocoding lookup unavailable (${err.message})`
      };
      task.resolve(errData);
    }
  }

  isGeocodingQueueRunning = false;
}

function initTripVisualization(markdownText, data) {
  const parsed = parseItineraryLocations(markdownText, data);
  const dayFilterBar = document.getElementById("dayFilterBar");
  const dayFilterPills = document.getElementById("dayFilterPills");
  const mapStatus = document.getElementById("mapStatus");
  const unmappedBox = document.getElementById("unmappedLocationsBox");
  const unmappedList = document.getElementById("unmappedLocationsList");

  if (!dayFilterBar || !dayFilterPills || !mapStatus) return;

  // Clear previous state
  dayFilterPills.innerHTML = "";
  unmappedList.innerHTML = "";
  unmappedBox.classList.add("hidden");
  mapStatus.classList.remove("hidden");
  mapStatus.textContent = "Loading map and pinpointing locations...";

  // Check if Leaflet is loaded
  if (typeof L === "undefined") {
    mapStatus.textContent = "Map library (Leaflet) is loading or unavailable.";
    return;
  }

  const mapContainer = document.getElementById("tripMap");
  if (!mapContainer) return;

  if (!tripMapInstance) {
    tripMapInstance = L.map("tripMap", {
      zoomControl: true,
      scrollWheelZoom: false
    });

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> contributors',
      maxZoom: 19
    }).addTo(tripMapInstance);

    tripMarkersGroup = L.featureGroup().addTo(tripMapInstance);
    tripMapInstance.setView([20, 0], 2);
  } else {
    tripMarkersGroup.clearLayers();
  }

  mapMarkersList = [];
  activeDayFilter = "all";

  // Force Leaflet container resize when section opens
  setTimeout(() => {
    if (tripMapInstance) tripMapInstance.invalidateSize();
  }, 100);
  setTimeout(() => {
    if (tripMapInstance) tripMapInstance.invalidateSize();
  }, 400);

  // Render Day Filter Pills
  if (parsed.days.length > 0 || parsed.locations.length > 0) {
    dayFilterBar.classList.remove("hidden");

    // All Days pill
    const allPill = document.createElement("button");
    allPill.className = "day-pill active";
    allPill.textContent = "All Days";
    allPill.onclick = () => filterMapByDay("all");
    dayFilterPills.appendChild(allPill);

    // Individual Day pills
    parsed.days.forEach((dayNum) => {
      const pill = document.createElement("button");
      pill.className = "day-pill";
      pill.textContent = `Day ${dayNum}`;
      pill.onclick = () => filterMapByDay(dayNum);
      dayFilterPills.appendChild(pill);
    });

    // Hotels pill
    const hasHotels = parsed.locations.some((l) => l.category === "hotel");
    if (hasHotels) {
      const hotelPill = document.createElement("button");
      hotelPill.className = "day-pill";
      hotelPill.textContent = "🏨 Hotels";
      hotelPill.onclick = () => filterMapByDay("hotels");
      dayFilterPills.appendChild(hotelPill);
    }

    // Dining pill
    const hasDining = parsed.locations.some((l) => l.category === "restaurant");
    if (hasDining) {
      const diningPill = document.createElement("button");
      diningPill.className = "day-pill";
      diningPill.textContent = "🍽️ Dining";
      diningPill.onclick = () => filterMapByDay("dining");
      dayFilterPills.appendChild(diningPill);
    }

    // Markets pill
    const hasMarkets = parsed.locations.some((l) => l.category === "market");
    if (hasMarkets) {
      const marketPill = document.createElement("button");
      marketPill.className = "day-pill";
      marketPill.textContent = "🛍️ Markets";
      marketPill.onclick = () => filterMapByDay("markets");
      dayFilterPills.appendChild(marketPill);
    }
  } else {
    dayFilterBar.classList.add("hidden");
  }

  const combinedUnmapped = [...parsed.unmapped];
  updateUnmappedUI(combinedUnmapped);

  if (parsed.locations.length === 0) {
    mapStatus.textContent = "No specific landmarks identified for mapping in this itinerary.";
    return;
  }

  // Geocode destination city to center the map immediately
  if (parsed.destination) {
    geocodeLocation({ name: parsed.destination, area: "", category: "destination" }, "").then((destGeo) => {
      if (destGeo && destGeo.found && tripMapInstance && mapMarkersList.length === 0) {
        tripMapInstance.setView([destGeo.lat, destGeo.lon], 11);
      }
    });
  }

  // Geocode and plot each location
  let mappedCount = 0;
  const totalToGeocode = parsed.locations.length;
  mapStatus.textContent = `Pinpointing ${totalToGeocode} trip location(s)...`;

  parsed.locations.forEach((item) => {
    geocodeLocation(item, parsed.destination).then((geo) => {
      if (geo && geo.found && !isNaN(geo.lat) && !isNaN(geo.lon)) {
        addMarkerToMap(item, geo.lat, geo.lon);
        mappedCount++;
      } else {
        combinedUnmapped.push({
          name: item.name,
          day: item.day,
          type: item.category,
          reason: geo.reason || "Coordinates not found on map"
        });
        updateUnmappedUI(combinedUnmapped);
      }

      const plottedMsg = `Plotted ${mappedCount} of ${totalToGeocode} location(s) on map.`;
      const unmappedNote = combinedUnmapped.length > 0 ? ` (${combinedUnmapped.length} unmapped)` : "";
      mapStatus.textContent = plottedMsg + unmappedNote;
    });
  });
}

function addMarkerToMap(item, lat, lon) {
  if (!tripMapInstance || !tripMarkersGroup) return;

  let iconChar = "📍";
  let pinClass = "marker-attraction";
  let tagClass = "tag-attraction";
  let tagLabel = `📍 Day ${item.day || ""} Attraction`;

  if (item.category === "hotel") {
    iconChar = "🏨";
    pinClass = "marker-hotel";
    tagClass = "tag-hotel";
    tagLabel = "🏨 Recommended Hotel";
  } else if (item.category === "restaurant") {
    iconChar = "🍽️";
    pinClass = "marker-restaurant";
    tagClass = "tag-restaurant";
    tagLabel = item.day ? `🍽️ Day ${item.day} Dining` : "🍽️ Dining";
  } else if (item.category === "market") {
    iconChar = "🛍️";
    pinClass = "marker-market";
    tagClass = "tag-market";
    tagLabel = item.day ? `🛍️ Day ${item.day} Market` : "🛍️ Market";
  } else {
    // Specific attraction icons
    const nameLow = item.name.toLowerCase();
    if (nameLow.includes("beach")) iconChar = "🏖️";
    else if (nameLow.includes("fort") || nameLow.includes("palace") || nameLow.includes("castle")) iconChar = "🏰";
    else if (nameLow.includes("temple") || nameLow.includes("church") || nameLow.includes("cathedral")) iconChar = "🏛️";
  }

  const iconHtml = `<div class="map-marker-pin ${pinClass}" title="${escapeHtml(item.name)}">${iconChar}</div>`;

  const customIcon = L.divIcon({
    html: iconHtml,
    className: "custom-leaflet-pin",
    iconSize: [32, 32],
    iconAnchor: [16, 16],
    popupAnchor: [0, -18]
  });

  const marker = L.marker([lat, lon], { icon: customIcon });

  const popupContent = `
    <div class="map-popup-card">
      <span class="popup-tag ${tagClass}">
        ${tagLabel}
      </span>
      <h4 class="popup-name">${escapeHtml(item.name)}</h4>
      ${item.area ? `<div class="popup-area" style="font-size:0.78rem;color:#94a3b8;font-weight:600;margin:2px 0;">📍 ${escapeHtml(item.area)}</div>` : ''}
      ${item.snippet ? `<p class="popup-snippet">${escapeHtml(item.snippet)}</p>` : ''}
    </div>
  `;

  marker.bindPopup(popupContent);
  tripMarkersGroup.addLayer(marker);

  mapMarkersList.push({
    marker: marker,
    data: item
  });

  try {
    const bounds = tripMarkersGroup.getBounds();
    if (bounds.isValid()) {
      tripMapInstance.fitBounds(bounds, { padding: [35, 35], maxZoom: 15 });
    }
  } catch (e) {}
}

function filterMapByDay(filterKey) {
  activeDayFilter = filterKey;

  // Update pills active class
  const pills = document.querySelectorAll(".day-pill");
  pills.forEach((p) => {
    if (filterKey === "all" && p.textContent === "All Days") p.classList.add("active");
    else if (filterKey === "hotels" && p.textContent.includes("Hotels")) p.classList.add("active");
    else if (filterKey === "dining" && p.textContent.includes("Dining")) p.classList.add("active");
    else if (filterKey === "markets" && p.textContent.includes("Markets")) p.classList.add("active");
    else if (p.textContent === `Day ${filterKey}`) p.classList.add("active");
    else p.classList.remove("active");
  });

  const activeLatLngs = [];

  mapMarkersList.forEach((entry) => {
    const markerEl = entry.marker.getElement();
    const pin = markerEl ? markerEl.querySelector(".map-marker-pin") : null;
    const cat = entry.data.category;
    const day = entry.data.day;

    let isVisible = false;
    if (filterKey === "all") {
      isVisible = true;
    } else if (filterKey === "hotels") {
      isVisible = (cat === "hotel");
    } else if (filterKey === "dining") {
      isVisible = (cat === "restaurant");
    } else if (filterKey === "markets") {
      isVisible = (cat === "market");
    } else {
      // Selected day number: show that day's places + keep hotels visible
      isVisible = (day === filterKey || cat === "hotel");
    }

    if (pin) {
      if (isVisible) {
        pin.classList.remove("marker-inactive");
        const isHighlightMatch = (
          (filterKey !== "all") && (
            (day === filterKey) ||
            (filterKey === "hotels" && cat === "hotel") ||
            (filterKey === "dining" && cat === "restaurant") ||
            (filterKey === "markets" && cat === "market")
          )
        );
        if (isHighlightMatch) {
          pin.classList.add("marker-highlighted");
        } else {
          pin.classList.remove("marker-highlighted");
        }
      } else {
        pin.classList.add("marker-inactive");
        pin.classList.remove("marker-highlighted");
      }
    }

    if (isVisible) {
      activeLatLngs.push(entry.marker.getLatLng());
    }
  });

  if (tripMapInstance && activeLatLngs.length > 0) {
    const bounds = L.latLngBounds(activeLatLngs);
    if (bounds.isValid()) {
      tripMapInstance.fitBounds(bounds, { padding: [40, 40], maxZoom: 15 });
    }
  }
}

function updateUnmappedUI(unmappedList) {
  const box = document.getElementById("unmappedLocationsBox");
  const listEl = document.getElementById("unmappedLocationsList");
  if (!box || !listEl) return;

  if (!unmappedList || unmappedList.length === 0) {
    box.classList.add("hidden");
    listEl.innerHTML = "";
    return;
  }

  box.classList.remove("hidden");
  listEl.innerHTML = "";
  unmappedList.forEach((item) => {
    const li = document.createElement("li");
    const dayLabel = item.day ? `Day ${item.day}: ` : "";
    li.innerHTML = `<strong>${dayLabel}${escapeHtml(item.name)}</strong> — <span>${escapeHtml(item.reason)}</span>`;
    listEl.appendChild(li);
  });
}

// Window resize listener to keep map viewport responsive
window.addEventListener("resize", () => {
  if (tripMapInstance) {
    tripMapInstance.invalidateSize();
  }
});