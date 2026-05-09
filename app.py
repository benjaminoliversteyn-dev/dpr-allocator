"""
Distributed Peer Review (DPR) Allocator
A Streamlit web app that replicates the R-based DPR allocation logic.
"""

import streamlit as st
import pandas as pd
import numpy as np
import random
import io
import zipfile
from copy import deepcopy

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DPR Allocator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Main header */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #1a1a2e;
    line-height: 1.15;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 1.05rem;
    color: #555;
    font-weight: 300;
    margin-bottom: 2rem;
}

/* Step cards */
.step-card {
    background: #f7f7fb;
    border-left: 4px solid #4f46e5;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.step-number {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4f46e5;
}
.step-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: #1a1a2e;
    margin: 0;
}

/* Metric boxes */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
    flex-wrap: wrap;
}
.metric-box {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: white;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    flex: 1;
    min-width: 130px;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    display: block;
}
.metric-label {
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    opacity: 0.85;
}

/* Status badges */
.badge-ok  { background:#d1fae5; color:#065f46; padding:3px 10px; border-radius:99px; font-size:0.8rem; font-weight:600; }
.badge-warn{ background:#fef3c7; color:#92400e; padding:3px 10px; border-radius:99px; font-size:0.8rem; font-weight:600; }
.badge-err { background:#fee2e2; color:#991b1b; padding:3px 10px; border-radius:99px; font-size:0.8rem; font-weight:600; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #1a1a2e;
}
section[data-testid="stSidebar"] * {
    color: #e2e2f0 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stTextInput label {
    color: #a0a0c0 !important;
    font-size: 0.8rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.55rem 1.4rem;
    transition: opacity 0.2s;
}
.stDownloadButton > button:hover { opacity: 0.88; }

/* Primary run button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 1rem;
    padding: 0.65rem 2rem;
    width: 100%;
}

hr { border-color: #e5e5ef; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Allocation logic (Python port of the R code)
# ─────────────────────────────────────────────

def parse_conflicts(conflict_str):
    """Return a list of conflict names from a comma-separated string."""
    if pd.isna(conflict_str) or str(conflict_str).strip() == "":
        return []
    return [c.strip() for c in str(conflict_str).split(",") if c.strip()]


def generate_constraint_list(data1: pd.DataFrame, data2: pd.DataFrame, master: pd.DataFrame) -> dict:
    """
    For each person in data1, build the set of IDs in data2 they CANNOT review.
    Constraints come from the 'Conflicts' column (names of conflicted people).
    """
    # Build name→ID and email→ID lookup from master
    name_to_id = dict(zip(master["ApplicantName"], master["ID"]))
    constraints = {}
    for _, row in data1.iterrows():
        forbidden_ids = set()
        conflicts = parse_conflicts(row.get("Conflicts", ""))
        for cname in conflicts:
            if cname in name_to_id:
                cid = name_to_id[cname]
                # Only add if the conflicted person is in data2
                if cid in data2["ID"].values:
                    forbidden_ids.add(cid)
        constraints[row["ID"]] = forbidden_ids
    return constraints


def dpr_allocation(group_to_allocate: pd.DataFrame,
                   group_from: pd.DataFrame,
                   constraint_list: dict,
                   reviews_per_person: int = 10) -> tuple:
    """
    Allocate `reviews_per_person` reviewees from group_from to each person in group_to_allocate.
    Returns (assignment_matrix as dict[reviewer_id → list[reviewee_ids]], count_dict).
    """
    from_ids = list(group_from["ID"])
    count = {fid: 0 for fid in from_ids}  # how many times each from-person has been assigned

    assignments = {}  # reviewer_id → [reviewee_ids]

    # Shuffle the order so allocation is random
    allocate_order = list(group_to_allocate["ID"])
    random.shuffle(allocate_order)

    for reviewer_id in allocate_order:
        forbidden = constraint_list.get(reviewer_id, set())
        # Available pool: not forbidden, count < reviews_per_person, not already assigned to this reviewer
        already_assigned = set(assignments.get(reviewer_id, []))

        available = [fid for fid in from_ids
                     if fid not in forbidden
                     and fid not in already_assigned
                     and count[fid] < reviews_per_person]

        random.shuffle(available)
        chosen = available[:reviews_per_person]

        assignments[reviewer_id] = chosen
        for c in chosen:
            count[c] += 1

    return assignments, count


def redistribute_incomplete(assignments: dict,
                             group_to_allocate: pd.DataFrame,
                             group_from: pd.DataFrame,
                             constraint_list: dict,
                             count: dict,
                             reviews_per_person: int = 10) -> tuple:
    """
    For any reviewer who has fewer than reviews_per_person assignees,
    try to find valid swaps or additions.
    """
    from_ids = set(group_from["ID"])
    max_iter = 500

    for _ in range(max_iter):
        # Find reviewers with too few
        short = [rid for rid, lst in assignments.items() if len(lst) < reviews_per_person]
        if not short:
            break

        for short_reviewer in short:
            needed = reviews_per_person - len(assignments[short_reviewer])
            forbidden = constraint_list.get(short_reviewer, set())
            already = set(assignments[short_reviewer])

            # Direct additions from under-assigned pool
            available = [fid for fid in from_ids
                         if fid not in forbidden
                         and fid not in already
                         and count.get(fid, 0) < reviews_per_person]
            random.shuffle(available)
            additions = available[:needed]
            for a in additions:
                assignments[short_reviewer].append(a)
                count[a] = count.get(a, 0) + 1

            # If still short, try swapping: find a reviewer with exactly 10,
            # whose assignee can be freed and given to short_reviewer
            still_needed = reviews_per_person - len(assignments[short_reviewer])
            if still_needed <= 0:
                continue

            for donor_reviewer, donor_list in assignments.items():
                if len(donor_list) < reviews_per_person:
                    continue
                for candidate in list(donor_list):
                    if (candidate not in forbidden
                            and candidate not in set(assignments[short_reviewer])):
                        # Check donor can afford to lose this person
                        # (donor must still be able to get a replacement)
                        donor_forbidden = constraint_list.get(donor_reviewer, set())
                        replacement_pool = [fid for fid in from_ids
                                            if fid not in donor_forbidden
                                            and fid not in set(donor_list)
                                            and fid != candidate
                                            and count.get(fid, 0) < reviews_per_person]
                        if not replacement_pool:
                            continue
                        replacement = random.choice(replacement_pool)
                        # Execute swap
                        donor_list.remove(candidate)
                        donor_list.append(replacement)
                        count[replacement] = count.get(replacement, 0) + 1
                        # candidate count stays the same (just moved)
                        assignments[short_reviewer].append(candidate)
                        still_needed -= 1
                        if still_needed <= 0:
                            break
                if still_needed <= 0:
                    break

    return assignments, count


def find_reciprocals(group1_assignments: dict, group2: pd.DataFrame) -> dict:
    """
    For each person in group2, find who in group1 has been assigned to review them
    (they should not review those people back).
    Returns dict: group2_id → set of group1_ids that reviewed them.
    """
    reciprocals = {row["ID"]: set() for _, row in group2.iterrows()}
    for reviewer_id, reviewee_list in group1_assignments.items():
        for reviewee_id in reviewee_list:
            if reviewee_id in reciprocals:
                reciprocals[reviewee_id].add(reviewer_id)
    return reciprocals


def build_final_df(assignments: dict, master: pd.DataFrame, reviews_per_person: int) -> pd.DataFrame:
    """Convert assignments dict to a readable DataFrame."""
    id_to_name = dict(zip(master["ID"], master["ApplicantName"]))
    id_to_email = dict(zip(master["ID"], master.get("Email", pd.Series(dtype=str))))
    id_to_appnum = dict(zip(master["ID"], master.get("ApplicationNumber", master["ID"])))

    rows = []
    for reviewer_id, reviewees in assignments.items():
        row = {
            "Reviewer_ID": reviewer_id,
            "Reviewer_Name": id_to_name.get(reviewer_id, str(reviewer_id)),
            "Reviewer_Email": id_to_email.get(reviewer_id, ""),
            "Reviewer_ApplicationNumber": id_to_appnum.get(reviewer_id, reviewer_id),
        }
        for i, rev_id in enumerate(reviewees, 1):
            row[f"Reviewee_{i}_Name"] = id_to_name.get(rev_id, str(rev_id))
            row[f"Reviewee_{i}_Email"] = id_to_email.get(rev_id, "")
            row[f"Reviewee_{i}_AppNum"] = id_to_appnum.get(rev_id, rev_id)
        # Pad with blanks if fewer than target
        for i in range(len(reviewees) + 1, reviews_per_person + 1):
            row[f"Reviewee_{i}_Name"] = ""
            row[f"Reviewee_{i}_Email"] = ""
            row[f"Reviewee_{i}_AppNum"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def run_allocation(master: pd.DataFrame, reviews_per_person: int, seed: int):
    """Full allocation pipeline."""
    random.seed(seed)
    np.random.seed(seed)

    ids = list(master["ID"])
    random.shuffle(ids)
    half = len(ids) // 2
    group1_ids = set(ids[:half])
    group2_ids = set(ids[half:])

    group1 = master[master["ID"].isin(group1_ids)].reset_index(drop=True)
    group2 = master[master["ID"].isin(group2_ids)].reset_index(drop=True)

    # ── Group 1 reviews Group 2 ──
    g1_constraints = generate_constraint_list(group1, group2, master)
    g1_assignments, g2_count = dpr_allocation(group1, group2, g1_constraints, reviews_per_person)
    g1_assignments, g2_count = redistribute_incomplete(g1_assignments, group1, group2, g1_constraints, g2_count, reviews_per_person)

    # ── Reciprocals: prevent group2 reviewing back ──
    reciprocals = find_reciprocals(g1_assignments, group2)

    # Add reciprocals into master conflicts for group2 constraint generation
    master2 = master.copy()
    id_to_name = dict(zip(master["ID"], master["ApplicantName"]))
    for g2_id, recip_ids in reciprocals.items():
        recip_names = [id_to_name.get(r, str(r)) for r in recip_ids]
        idx = master2[master2["ID"] == g2_id].index
        if len(idx):
            existing = master2.loc[idx[0], "Conflicts"]
            existing_list = parse_conflicts(existing)
            combined = list(set(existing_list + recip_names))
            master2.loc[idx[0], "Conflicts"] = ",".join(combined)

    # ── Group 2 reviews Group 1 ──
    g2_constraints = generate_constraint_list(group2, group1, master2)
    g2_assignments, g1_count = dpr_allocation(group2, group1, g2_constraints, reviews_per_person)
    g2_assignments, g1_count = redistribute_incomplete(g2_assignments, group2, group1, g2_constraints, g1_count, reviews_per_person)

    # ── Build output DataFrames ──
    df_g1 = build_final_df(g1_assignments, master, reviews_per_person)
    df_g2 = build_final_df(g2_assignments, master, reviews_per_person)
    df_combined = pd.concat([df_g1, df_g2], ignore_index=True)

    # ── Summary stats ──
    all_assignments = {**g1_assignments, **g2_assignments}
    review_counts = {}
    for reviewer_id, reviewees in all_assignments.items():
        for rid in reviewees:
            review_counts[rid] = review_counts.get(rid, 0) + 1

    stats = {
        "total_applicants": len(master),
        "group1_size": len(group1),
        "group2_size": len(group2),
        "full_reviews": sum(1 for c in review_counts.values() if c >= reviews_per_person),
        "partial_reviews": sum(1 for c in review_counts.values() if c < reviews_per_person),
        "reviewers_full": sum(1 for lst in all_assignments.values() if len(lst) >= reviews_per_person),
        "reviewers_short": sum(1 for lst in all_assignments.values() if len(lst) < reviews_per_person),
        "review_count_distribution": review_counts,
    }

    return df_combined, df_g1, df_g2, stats, master2


# ─────────────────────────────────────────────
# Template CSV
# ─────────────────────────────────────────────
def make_template_csv():
    names = [
        "Alice Smith", "Bob Jones", "Carol White", "David Lee", "Eve Brown",
        "Frank Miller", "Grace Wilson", "Henry Moore", "Isla Taylor", "Jack Anderson",
        "Karen Thomas", "Liam Jackson", "Maya Harris", "Noah Martin", "Olivia Thompson",
        "Peter Garcia", "Quinn Martinez", "Rachel Robinson", "Sam Clark", "Tara Lewis",
        "Uma Walker", "Victor Hall", "Wendy Allen", "Xavier Young", "Yara King",
        "Zara Wright", "Aaron Scott", "Beth Green", "Carlos Adams", "Diana Baker",
        "Ethan Nelson", "Fiona Carter", "George Mitchell", "Hannah Perez", "Ivan Roberts",
        "Julia Turner", "Kevin Phillips", "Laura Campbell", "Marcus Parker", "Nina Evans",
        "Oscar Edwards", "Priya Collins", "Quinn Stewart", "Rosa Sanchez", "Samuel Morris",
        "Tina Rogers", "Ulrich Reed", "Vera Cook", "William Morgan", "Xena Bell",
    ]
    # A handful of realistic conflicts (by name), rest empty
    conflicts = [""] * 50
    conflicts[0]  = "Bob Jones"
    conflicts[1]  = "Alice Smith"
    conflicts[4]  = "Frank Miller"
    conflicts[5]  = "Eve Brown"
    conflicts[10] = "Liam Jackson"
    conflicts[11] = "Karen Thomas"
    conflicts[20] = "Victor Hall"
    conflicts[21] = "Uma Walker"
    conflicts[30] = "Fiona Carter"
    conflicts[31] = "Ethan Nelson"
    conflicts[40] = "Priya Collins"
    conflicts[41] = "Oscar Edwards"

    template = pd.DataFrame({
        "ID": list(range(1, 51)),
        "ApplicantName": names,
        "Email": [f"{n.split()[0].lower()}.{n.split()[1].lower()}@example.com" for n in names],
        "ApplicationNumber": [f"APP{str(i).zfill(3)}" for i in range(1, 51)],
        "Conflicts": conflicts,
    })
    return template.to_csv(index=False)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    reviews_per_person = st.number_input(
        "Reviews per person",
        min_value=1, max_value=30, value=10,
        help="How many applications each person reviews (and receives reviews from)."
    )

    seed_col, info_col = st.columns([5, 1])
    with seed_col:
        random_seed = st.number_input(
            "Random seed",
            min_value=0, max_value=99999, value=122,
        )
    with info_col:
        st.markdown("<div style='margin-top:1.9rem'></div>", unsafe_allow_html=True)
        with st.popover("ℹ️"):
            st.markdown("""
**What is a random seed?**

The allocation involves lots of random choices — which group each person goes into, who gets assigned to whom, etc.

The seed is a "starting point" for all that randomness:

- **Same seed = same result every time.**  
  Running the allocation twice with seed `122` gives identical groups and assignments. Useful for reproducibility.

- **Different seed = different allocation.**  
  Not happy with a result? Change the seed to any other number and rerun — you'll get a completely different but equally valid allocation.

The default `122` matches the original script.
            """)

    st.markdown("---")
    st.markdown("### 📋 Required CSV columns")
    for col, desc in [
        ("**ID**", "Unique integer per applicant"),
        ("**ApplicantName**", "Full name"),
        ("**Email**", "Email address"),
        ("**ApplicationNumber**", "Reference code"),
        ("**Conflicts**", "Comma-separated names of conflicted people (can be empty)"),
    ]:
        st.markdown(f"{col} — {desc}")

    st.markdown("---")
    st.download_button(
        "⬇️ Download template CSV",
        data=make_template_csv(),
        file_name="dpr_template.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────
st.markdown('<p class="hero-title">Distributed Peer Review<br>Allocator 🔬</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Upload your applicants, set your preferences, and generate a fair reviewer allocation in seconds.</p>', unsafe_allow_html=True)

# How it works
with st.expander("ℹ️  How does this work?", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="step-card">
            <span class="step-number">Step 1</span>
            <p class="step-title">Split the pool</p>
            <p style="font-size:0.88rem;color:#555;margin-top:0.4rem">
            All applicants are randomly split into two equal groups.
            </p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="step-card">
            <span class="step-number">Step 2</span>
            <p class="step-title">Cross-group review</p>
            <p style="font-size:0.88rem;color:#555;margin-top:0.4rem">
            Group 1 reviews Group 2 and vice versa. No-one reviews their own group.
            </p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="step-card">
            <span class="step-number">Step 3</span>
            <p class="step-title">Respect conflicts</p>
            <p style="font-size:0.88rem;color:#555;margin-top:0.4rem">
            Declared conflicts and reciprocal reviews are never paired together.
            </p>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── File upload ──
st.subheader("📂 Upload your applicant data")
uploaded_file = st.file_uploader(
    "Upload a CSV file (see sidebar for required columns and template)",
    type=["csv"],
    label_visibility="collapsed",
)

if uploaded_file:
    try:
        master_df = pd.read_csv(uploaded_file)

        # Validate columns
        required_cols = {"ID", "ApplicantName", "Email"}
        missing = required_cols - set(master_df.columns)
        if missing:
            st.error(f"❌ Your CSV is missing these required columns: **{', '.join(missing)}**")
            st.stop()

        # Add optional columns if missing
        if "ApplicationNumber" not in master_df.columns:
            master_df["ApplicationNumber"] = master_df["ID"]
        if "Conflicts" not in master_df.columns:
            master_df["Conflicts"] = ""

        master_df["ID"] = master_df["ID"].astype(int)

        # Preview
        st.markdown("**Preview of uploaded data:**")
        st.dataframe(master_df.head(10), use_container_width=True)

        n = len(master_df)
        min_needed = reviews_per_person * 2 + 1
        if n < min_needed:
            st.warning(
                f"⚠️ You have **{n} applicants** but need at least **{min_needed}** "
                f"for everyone to receive {reviews_per_person} reviewers. "
                "Results may be incomplete — consider reducing *Reviews per person* in the sidebar."
            )

        st.markdown(f"✅ **{n} applicants** loaded successfully.")
        st.markdown("---")

        # ── Run allocation ──
        if st.button("🚀 Run Allocation", type="primary"):
            with st.spinner("Running allocation — this may take a moment for large datasets..."):
                try:
                    df_combined, df_g1, df_g2, stats, _ = run_allocation(
                        master_df.copy(), reviews_per_person, int(random_seed)
                    )

                    st.success("✅ Allocation complete!")
                    st.markdown("---")

                    # ── Metrics ──
                    st.subheader("📊 Results summary")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total applicants", stats["total_applicants"])
                    col2.metric("Group 1 size", stats["group1_size"])
                    col3.metric("Group 2 size", stats["group2_size"])
                    col4.metric("Got full reviews", stats["full_reviews"])

                    # Review count distribution
                    dist = stats["review_count_distribution"]
                    if dist:
                        min_r = min(dist.values())
                        max_r = max(dist.values())
                        avg_r = sum(dist.values()) / len(dist)

                        colA, colB, colC = st.columns(3)
                        colA.metric("Min reviews received", min_r,
                                    delta=f"{min_r - reviews_per_person} vs target",
                                    delta_color="normal")
                        colB.metric("Avg reviews received", f"{avg_r:.1f}")
                        colC.metric("Max reviews received", max_r)

                    if stats["partial_reviews"] > 0:
                        st.warning(
                            f"⚠️ **{stats['partial_reviews']} applicant(s)** received fewer than "
                            f"{reviews_per_person} reviews. This can happen when conflict constraints "
                            "are very restrictive. Try a different seed or review your conflict list."
                        )

                    # ── Preview table ──
                    st.markdown("---")
                    st.subheader("🗂️ Allocation preview (first 20 rows)")
                    preview_cols = ["Reviewer_Name", "Reviewer_Email", "Reviewer_ApplicationNumber"] + \
                                   [c for c in df_combined.columns if c.startswith("Reviewee_") and "_Name" in c]
                    st.dataframe(df_combined[preview_cols].head(20), use_container_width=True)

                    # ── Downloads ──
                    st.markdown("---")
                    st.subheader("⬇️ Download results")

                    def to_csv(df):
                        return df.to_csv(index=False).encode("utf-8")

                    # Zip of all files
                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        zf.writestr("dpr_all_allocations.csv", df_combined.to_csv(index=False))
                        zf.writestr("dpr_group1_allocations.csv", df_g1.to_csv(index=False))
                        zf.writestr("dpr_group2_allocations.csv", df_g2.to_csv(index=False))
                    zip_buf.seek(0)

                    colD, colE, colF = st.columns(3)
                    with colD:
                        st.download_button(
                            "📦 Download all (ZIP)",
                            data=zip_buf.getvalue(),
                            file_name="dpr_results.zip",
                            mime="application/zip",
                        )
                    with colE:
                        st.download_button(
                            "📄 Group 1 CSV",
                            data=to_csv(df_g1),
                            file_name="dpr_group1_allocations.csv",
                            mime="text/csv",
                        )
                    with colF:
                        st.download_button(
                            "📄 Group 2 CSV",
                            data=to_csv(df_g2),
                            file_name="dpr_group2_allocations.csv",
                            mime="text/csv",
                        )

                    st.markdown(
                        "<p style='font-size:0.82rem;color:#888;margin-top:1rem'>"
                        "Each CSV row is one reviewer. Columns show their assigned reviewees' "
                        "names, emails, and application numbers."
                        "</p>",
                        unsafe_allow_html=True,
                    )

                except Exception as e:
                    st.error(f"❌ Allocation failed: {e}")
                    st.exception(e)

    except Exception as e:
        st.error(f"❌ Could not read your CSV: {e}")

else:
    # Empty state illustration
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;background:#f7f7fb;border-radius:12px;border:2px dashed #c7c7e0;">
        <span style="font-size:3rem">📎</span>
        <p style="font-family:'DM Serif Display',serif;font-size:1.4rem;color:#1a1a2e;margin:0.5rem 0">
            Drop your CSV here to get started
        </p>
        <p style="color:#888;font-size:0.9rem;">
            Download the template from the sidebar if you need a starting point.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='font-size:0.78rem;color:#aaa;text-align:center;'>"
    "Based on the DPR allocation algorithm by Cillian Brophy · "
    "Groups are split randomly, conflicts respected, reviews balanced."
    "</p>",
    unsafe_allow_html=True,
)
