const API_BASE = "http://127.0.0.1:8001";

const MAX_UPLOAD_MB = 15;

const ALLOWED_TYPES = [
  "image/jpeg",
  "image/png",
  "image/webp",
  "image/bmp"
];

const $ = (id) => document.getElementById(id);


/* ============================================================
   NAVIGATION
   ============================================================ */

document.querySelectorAll(".tab").forEach((button) => {

  button.addEventListener("click", () => {
    switchView(button.dataset.view);
  });

});


function switchView(view) {

  document.querySelectorAll(".tab").forEach((button) => {

    button.classList.toggle(
      "active",
      button.dataset.view === view
    );

  });


  document.querySelectorAll(".view").forEach((section) => {

    section.classList.toggle(
      "active",
      section.id === `view-${view}`
    );

  });


  if (view === "history") {
    loadHistory();
  }

}


/* ============================================================
   UPLOAD ELEMENTS
   ============================================================ */

const dropzone = $("dropzone");
const fileInput = $("fileInput");

const dzEmpty = $("dzEmpty");
const previewWrap = $("previewWrap");
const preview = $("preview");

const chooseBtn = $("chooseBtn");
const changeBtn = $("changeBtn");

const analyzeBtn = $("analyzeBtn");

const uploadError = $("uploadError");
const fileName = $("fileName");


let selectedFile = null;
let previewUrl = null;


/* ============================================================
   FILE PICKER
   ============================================================ */

function openFilePicker(event) {

  if (event) {
    event.preventDefault();
    event.stopPropagation();
  }

  fileInput.click();

}


chooseBtn.addEventListener(
  "click",
  openFilePicker
);


changeBtn.addEventListener(
  "click",
  openFilePicker
);


/*
 * Clicking anywhere in the upload area opens
 * the file picker.
 *
 * Buttons are excluded because they already
 * have their own click handler.
 */

dropzone.addEventListener(
  "click",
  (event) => {

    if (
      event.target === chooseBtn ||
      chooseBtn.contains(event.target) ||
      event.target === changeBtn ||
      changeBtn.contains(event.target)
    ) {
      return;
    }

    fileInput.click();

  }
);


/* ============================================================
   KEYBOARD UPLOAD
   ============================================================ */

dropzone.addEventListener(
  "keydown",
  (event) => {

    if (
      event.key === "Enter" ||
      event.key === " "
    ) {

      event.preventDefault();

      fileInput.click();

    }

  }
);


/* ============================================================
   FILE SELECTION
   ============================================================ */

fileInput.addEventListener(
  "change",
  (event) => {

    const file = event.target.files[0];

    handleFile(file);

    /*
     * Allows selecting the same file again.
     */

    event.target.value = "";

  }
);


/* ============================================================
   DRAG & DROP
   ============================================================ */

["dragenter", "dragover"].forEach(
  (eventName) => {

    dropzone.addEventListener(
      eventName,
      (event) => {

        event.preventDefault();

        event.stopPropagation();

        dropzone.classList.add("drag");

      }
    );

  }
);


["dragleave", "drop"].forEach(
  (eventName) => {

    dropzone.addEventListener(
      eventName,
      (event) => {

        event.preventDefault();

        event.stopPropagation();

        dropzone.classList.remove("drag");

      }
    );

  }
);


dropzone.addEventListener(
  "drop",
  (event) => {

    if (
      event.dataTransfer &&
      event.dataTransfer.files.length
    ) {

      handleFile(
        event.dataTransfer.files[0]
      );

    }

  }
);


/* ============================================================
   HANDLE FILE
   ============================================================ */

function handleFile(file) {

  if (!file) {
    return;
  }

  hideUploadError();


  /* File type */

  if (!ALLOWED_TYPES.includes(file.type)) {

    showUploadError(
      "Unsupported file type. Please choose JPG, PNG, WEBP, or BMP."
    );

    return;

  }


  /* File size */

  if (
    file.size >
    MAX_UPLOAD_MB * 1024 * 1024
  ) {

    showUploadError(
      `File exceeds the ${MAX_UPLOAD_MB} MB limit.`
    );

    return;

  }


  selectedFile = file;


  /* Revoke old preview */

  if (previewUrl) {

    URL.revokeObjectURL(previewUrl);

  }


  previewUrl =
    URL.createObjectURL(file);


  preview.src = previewUrl;


  fileName.textContent =
    file.name;


  /* Switch upload area to preview */

  dzEmpty.hidden = true;

  previewWrap.hidden = false;


  /* Enable analysis */

  analyzeBtn.disabled = false;


  /* Reset previous result */

  resetResultPanel();

}


/* ============================================================
   ERROR
   ============================================================ */

function showUploadError(message) {

  uploadError.textContent = message;

  uploadError.hidden = false;

}


function hideUploadError() {

  uploadError.hidden = true;

  uploadError.textContent = "";

}


/* ============================================================
   RESULT ELEMENTS
   ============================================================ */

const resultEmpty =
  $("resultEmpty");

const resultLoading =
  $("resultLoading");

const resultContent =
  $("resultContent");

const resultStatus =
  $("resultStatus");

const resultPanel =
  $("resultPanel");


/* ============================================================
   RESET RESULT
   ============================================================ */

function resetResultPanel() {

  resultEmpty.hidden = false;

  resultLoading.hidden = true;

  resultContent.hidden = true;

  resultStatus.hidden = true;

  resultPanel.classList.remove(
    "has-result"
  );

}


/* ============================================================
   ANALYZE
   ============================================================ */

analyzeBtn.addEventListener(
  "click",
  async () => {

    if (!selectedFile) {
      return;
    }


    hideUploadError();


    /* Loading */

    resultEmpty.hidden = true;

    resultContent.hidden = true;

    resultStatus.hidden = true;

    resultLoading.hidden = false;

    resultPanel.classList.remove(
      "has-result"
    );


    analyzeBtn.disabled = true;

    const originalText =
      analyzeBtn.textContent;

    analyzeBtn.textContent =
      "Analyzing…";


    const form =
      new FormData();

    form.append(
      "file",
      selectedFile
    );


    try {

      const response =
        await fetch(
          `${API_BASE}/api/analyze`,
          {
            method: "POST",
            body: form
          }
        );


      const data =
        await response.json();


      if (!response.ok) {

        throw new Error(
          data.detail ||
          `Request failed with status ${response.status}`
        );

      }


      renderResult(data);


      /*
       * Update history after
       * successful analysis.
       */

      loadHistory();


    } catch (error) {

      console.error(
        "Analysis error:",
        error
      );


      resultLoading.hidden = true;

      resultContent.hidden = true;

      resultEmpty.hidden = true;

      resultStatus.hidden = true;

      resultPanel.classList.remove(
        "has-result"
      );


      showUploadError(
        error.message ||
        "Analysis failed. Please check that the backend is running."
      );


    } finally {

      analyzeBtn.disabled = false;

      analyzeBtn.textContent =
        originalText;

    }

  }
);


/* ============================================================
   RESULT RENDERING
   ============================================================ */

const DIAL_CIRCUMFERENCE =
  2 * Math.PI * 52;


function renderResult(data) {

  /*
   * IMPORTANT:
   *
   * Remove empty/loading states BEFORE
   * showing the result.
   */

  resultEmpty.hidden = true;

  resultLoading.hidden = true;

  resultContent.hidden = false;

  resultStatus.hidden = false;

  resultPanel.classList.add(
    "has-result"
  );


  /* ========================================================
     SCORE
     ======================================================== */

  const score =
    Number(data.quality_score) || 0;


  $("scoreNum").textContent =
    score.toFixed(0);


  const safeScore =
    Math.max(
      0,
      Math.min(
        100,
        score
      )
    );


  const dialArc =
    $("dialArc");


  dialArc.style.strokeDashoffset =
    DIAL_CIRCUMFERENCE *
    (1 - safeScore / 100);


  dialArc.style.stroke =
    labelColor(
      data.quality_label
    );


  /* ========================================================
     QUALITY LABEL
     ======================================================== */

  const qualityLabel =
    data.quality_label ||
    "UNKNOWN";


  const label =
    $("qualityLabel");


  label.textContent =
    qualityLabel;


  label.className =
    `quality-pill pill-${qualityLabel}`;


  /* ========================================================
     PRIMARY ISSUE
     ======================================================== */

  const firstIssue =
    data.issues &&
    data.issues.length
      ? data.issues[0]
      : null;


  const primaryIssue =
    $("primaryIssue");


  if (firstIssue) {

    primaryIssue.hidden = false;

    primaryIssue.textContent =
      `${capitalize(firstIssue.type)} detected`;

  } else {

    primaryIssue.hidden = true;

    primaryIssue.textContent = "";

  }


  /* ========================================================
     MODEL
     ======================================================== */

  $("modelVersion").textContent =
    `model: ${data.model_version || "—"}`;


  /* ========================================================
     DETAILS
     ======================================================== */

  renderIssues(
    data.issues || []
  );


  renderStats(
    data.stats || {}
  );

}


/* ============================================================
   ISSUES
   ============================================================ */

function renderIssues(issues) {

  const list =
    $("issuesList");


  list.innerHTML = "";


  if (!issues.length) {

    list.innerHTML = `
      <p class="no-issues">
        ✓ No issues detected — image looks clean.
      </p>
    `;

    return;

  }


  issues.forEach(
    (issue) => {

      const row =
        document.createElement(
          "div"
        );

      row.className =
        "issue-row";


      const dot =
        document.createElement(
          "span"
        );

      dot.className =
        `issue-dot dot-${issue.severity || "medium"}`;


      const type =
        document.createElement(
          "span"
        );

      type.className =
        "issue-type";

      type.textContent =
        capitalize(
          issue.type ||
          "Unknown issue"
        );


      const severity =
        document.createElement(
          "span"
        );

      severity.className =
        "issue-sev";

      severity.textContent =
        issue.severity ||
        "—";


      const confidence =
        document.createElement(
          "span"
        );

      confidence.className =
        "issue-conf";

      confidence.textContent =
        `${Math.round(
          Number(
            issue.confidence || 0
          ) * 100
        )}%`;


      row.append(
        dot,
        type,
        severity,
        confidence
      );


      list.appendChild(row);

    }
  );

}


/* ============================================================
   STATISTICS
   ============================================================ */

function renderStats(stats) {

  const grid =
    $("statsGrid");


  grid.innerHTML = "";


  const skip =
    new Set([
      "cnn_class_probs"
    ]);


  Object.entries(stats).forEach(
    ([key, value]) => {

      if (skip.has(key)) {
        return;
      }


      const cell =
        document.createElement(
          "div"
        );

      cell.className =
        "stat-cell";


      const label =
        document.createElement(
          "span"
        );

      label.className =
        "stat-key";

      label.textContent =
        key.replace(
          /_/g,
          " "
        );


      const val =
        document.createElement(
          "span"
        );

      val.className =
        `stat-val ${statClass(
          key,
          value
        )}`;


      val.textContent =
        formatStat(
          key,
          value
        );


      cell.append(
        label,
        val
      );


      grid.appendChild(
        cell
      );

    }
  );

}


/* ============================================================
   FORMAT STAT
   ============================================================ */

function formatStat(
  key,
  value
) {

  if (
    typeof value !== "number" ||
    !Number.isFinite(value)
  ) {

    return String(value);

  }


  const percentageKeys =
    new Set([
      "overexposed_fraction",
      "underexposed_fraction",
      "flat_region_fraction"
    ]);


  if (
    percentageKeys.has(key)
  ) {

    return `${
      (value * 100).toFixed(1)
    }%`;

  }


  if (
    Number.isInteger(value)
  ) {

    return String(value);

  }


  return value.toFixed(2);

}


/* ============================================================
   STAT CLASS
   ============================================================ */

function statClass(
  key,
  value
) {

  if (
    key === "overexposed_fraction" &&
    Number(value) > 0.05
  ) {

    return "warn";

  }


  if (
    key === "underexposed_fraction" &&
    Number(value) > 0.05
  ) {

    return "warn";

  }


  if (
    key === "degenerate_size" &&
    value === true
  ) {

    return "bad";

  }


  if (
    key === "cnn_agrees_with_rules" &&
    value === true
  ) {

    return "good";

  }


  if (
    key === "cnn_predicted_class" &&
    value !== "clean"
  ) {

    return "warn";

  }


  return "";

}


/* ============================================================
   HELPERS
   ============================================================ */

function labelColor(label) {

  if (label === "ACCEPTABLE") {
    return "var(--green)";
  }


  if (label === "DEGRADED") {
    return "var(--amber)";
  }


  return "var(--red)";

}


function capitalize(value) {

  return String(value)
    .replace(
      /_/g,
      " "
    )
    .replace(
      /\b\w/g,
      (char) =>
        char.toUpperCase()
    );

}


/* ============================================================
   HISTORY
   ============================================================ */

$("refreshHistory")
  .addEventListener(
    "click",
    loadHistory
  );


async function loadHistory() {

  try {

    const response =
      await fetch(
        `${API_BASE}/api/results?limit=100`
      );


    const rows =
      await response.json();


    if (!response.ok) {

      throw new Error(
        "History request failed"
      );

    }


    renderHistoryTable(
      rows
    );


  } catch (error) {

    console.error(
      "History error:",
      error
    );


    showHistoryError();

  }

}


/* ============================================================
   HISTORY TABLE
   ============================================================ */

function renderHistoryTable(
  rows
) {

  const body =
    $("historyBody");

  const wrap =
    $("historyTableWrap");

  const empty =
    $("historyEmpty");


  body.innerHTML = "";


  if (
    !rows ||
    !rows.length
  ) {

    wrap.hidden = true;

    empty.hidden = false;

    return;

  }


  empty.hidden = true;

  wrap.hidden = false;


  rows.forEach(
    (row) => {

      const tr =
        document.createElement(
          "tr"
        );


      /* IMAGE */

      const imageCell =
        document.createElement(
          "td"
        );


      const image =
        document.createElement(
          "img"
        );


      image.className =
        "history-image";

      image.alt = "";

      image.src =
        `${API_BASE}/api/results/${row.id}/image`;


      const name =
        document.createElement(
          "span"
        );


      name.className =
        "file-name";

      name.title =
        row.filename;

      name.textContent =
        row.filename;


      imageCell.append(
        image,
        name
      );


      /* SCORE */

      const scoreCell =
        document.createElement(
          "td"
        );


      scoreCell.textContent =
        Number(
          row.quality_score
        ).toFixed(1);


      /* LABEL */

      const labelCell =
        document.createElement(
          "td"
        );


      const pill =
        document.createElement(
          "span"
        );


      pill.className =
        `quality-pill pill-${row.quality_label}`;

      pill.textContent =
        row.quality_label;


      labelCell.appendChild(
        pill
      );


      /* DATE */

      const dateCell =
        document.createElement(
          "td"
        );


      dateCell.textContent =
        formatDate(
          row.created_at
        );


      /* ACTION */

      const actionCell =
        document.createElement(
          "td"
        );


      const viewButton =
        document.createElement(
          "button"
        );


      viewButton.type =
        "button";

      viewButton.className =
        "view-btn";

      viewButton.textContent =
        "View";


      viewButton.addEventListener(
        "click",
        () =>
          loadHistoricalResult(
            row.id
          )
      );


      actionCell.appendChild(
        viewButton
      );


      tr.append(
        imageCell,
        scoreCell,
        labelCell,
        dateCell,
        actionCell
      );


      body.appendChild(
        tr
      );

    }
  );

}


/* ============================================================
   LOAD HISTORICAL RESULT
   ============================================================ */

async function loadHistoricalResult(
  id
) {

  try {

    const response =
      await fetch(
        `${API_BASE}/api/results/${id}`
      );


    const data =
      await response.json();


    if (!response.ok) {

      throw new Error(
        data.detail ||
        "Could not load result"
      );

    }


    renderResult(
      data
    );


    switchView(
      "inspect"
    );


    window.scrollTo({
      top: 0,
      behavior: "smooth"
    });


  } catch (error) {

    showUploadError(
      error.message ||
      "Could not load this result."
    );

  }

}


/* ============================================================
   HISTORY ERROR
   ============================================================ */

function showHistoryError() {

  const empty =
    $("historyEmpty");


  empty.hidden = false;


  const strong =
    empty.querySelector(
      "strong"
    );


  const paragraph =
    empty.querySelector(
      "p"
    );


  if (strong) {

    strong.textContent =
      "Could not load history.";

  }


  if (paragraph) {

    paragraph.textContent =
      "Check that the backend is running, then try Refresh.";

  }


  $("historyTableWrap").hidden =
    true;

}


/* ============================================================
   DATE
   ============================================================ */

function formatDate(value) {

  const date =
    new Date(value);


  if (
    Number.isNaN(
      date.getTime()
    )
  ) {

    return String(value);

  }


  return date.toLocaleString(
    undefined,
    {
      dateStyle: "short",
      timeStyle: "short"
    }
  );

}


/* ============================================================
   INITIAL LOAD
   ============================================================ */

loadHistory();