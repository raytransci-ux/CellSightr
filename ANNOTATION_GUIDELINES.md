# Annotation Guidelines — Hemocytometer Cell Counter

**Hemocytometer:** Improved Neubauer
**Staining:** Trypan Blue
**Counting Region:** One corner square per image (4×4 = 16 sub-squares, 1mm × 1mm)
**Annotator:** Solo
**Tool:** Label Studio

---

## What Each Image Contains

Every image shows exactly one corner square of the Improved Neubauer hemocytometer:

```
┌────┬────┬────┬────┐
│    │    │    │    │
├────┼────┼────┼────┤   Total area:  1mm × 1mm
│    │    │    │    │   Depth:       0.1mm
├────┼────┼────┼────┤   Volume:      0.0001 mL (1 × 10⁻⁴ mL)
│    │    │    │    │
├────┼────┼────┼────┤   Concentration formula:
│    │    │    │    │   cells/mL = count × 10,000 × dilution factor
└────┴────┴────┴────┘
```

The 16 internal grid lines are **reference lines only** — they do NOT define separate
counting areas for annotation purposes. Annotate cells regardless of which sub-square
they fall in.

---

## Cell Labels

| Label | Color in Tool | Visual Appearance |
|---|---|---|
| `viable` | Green | Clear, bright, refractile cytoplasm — NO blue staining |
| `non_viable` | Blue | Dark blue or purple from trypan uptake — stained throughout |
| `ambiguous` | Yellow | Faint/partial blue staining, uncertain viability |

---

## Rule 1: What to Annotate

### Include (draw a bounding box):
- Every cell clearly visible inside the grid, regardless of which sub-square it's in
- Cells touching or overlapping any internal grid line
- Cells touching the outer boundary lines of the corner square
- Cells that are >50% visible at the edge of the image
- Dead cells (non_viable) — annotate these too, always

### Exclude (do NOT annotate):
- Cells that are <50% cut off at the image edge
- Debris, dust, or non-circular artifacts (unless clearly a cell)
- Out-of-focus blobs that cannot be identified as cells
- Air bubbles

---

## Rule 2: Drawing Bounding Boxes

- Draw the box **tightly** around the cell — just enough to enclose it
- Include the full cell boundary (halo/membrane) in the box
- The box should be **roughly square** (cells are circular/oval)
- For cells touching grid lines, the grid line can be inside the box

```
Good box:          Bad box (too loose):
 ┌──────┐           ┌──────────────┐
 │  ⬤   │           │              │
 └──────┘           │      ⬤       │
  Tight             │              │
                    └──────────────┘
```

---

## Rule 3: Overlapping and Clumped Cells

| Situation | Action |
|---|---|
| Two cells touching but boundaries visible | Draw two separate boxes |
| Cluster where individual cells cannot be separated | Draw one box around the cluster; label as the dominant type |
| One cell clearly on top of another | Annotate both if you can see two distinct cell outlines |

---

## Rule 4: Viability Classification

Use trypan blue uptake as the guide:

**`viable` (live):**
- Cytoplasm appears clear, bright, or slightly refractile
- No blue staining visible
- Cell membrane appears intact

**`non_viable` (dead):**
- Cell interior is dark blue or purple
- Stain is uniform throughout the cell
- May appear slightly larger or more irregular

**`ambiguous` (uncertain):**
- Very faint blue tinge — cannot confirm uptake
- Cell is out-of-focus but present
- Membrane appears compromised but staining unclear
- Use sparingly — these are filtered out during training

---

## Rule 5: Boundary Cells (Image Edges)

During annotation, treat image edges the same as the interior — annotate all visible cells.

**Note:** The counting boundary rule (count top/left, exclude bottom/right) is applied
automatically by the pipeline during inference. Do NOT try to apply this rule manually
during annotation.

---

## Annotation Workflow (Per Image)

1. **Scan systematically** — go sub-square by sub-square (left to right, top to bottom)
2. **First pass:** annotate all cells as `viable`
3. **Second pass:** go back and reclassify any blue-stained cells as `non_viable`
4. **Edge check:** confirm partial cells at image borders are included if >50% visible
5. **Quality check:** does your total count seem reasonable for the density?

---

## Time Target

- **Target:** < 5 minutes per image
- Dense images (>30 cells) may take up to 8 minutes
- Empty grids should take < 1 minute

---

## Metadata to Record per Image

Fill in `data/metadata_template.csv` for each image:

| Field | Example Values | Notes |
|---|---|---|
| `filename` | `img_001.jpg` | Exact filename |
| `density` | `empty`, `low`, `medium`, `high` | empty=0, low=1-10, medium=11-30, high=30+ |
| `has_trypan` | `yes` / `no` | Whether trypan blue staining was applied |
| `grid_visible` | `yes` / `partial` / `no` | Can you see all 16 sub-squares? |
| `focus_quality` | `good` / `acceptable` / `poor` | `poor` = exclude from training |
| `dilution_factor` | `2`, `10`, etc. | Cell suspension dilution before loading |
| `acquisition_date` | `2026-02-15` | Date of image capture |
| `notes` | Optional | Edge cases, unusual features |

---

## Pilot Phase (First 20 Images)

For your first 20 images:
1. Annotate carefully without rushing
2. After every 5 images, pause and review — are your boxes consistent in size?
3. Note any edge cases or situations not covered by these guidelines
4. Estimate your average time per image

After 20 images, review a few early annotations and verify they still look correct.
Consistency is more important than speed.

---

## Quality Checkpoints

**After every 25 images:**
- Re-open 3 images from earlier in the session
- Verify your labeling is still consistent
- Check that box sizes are appropriate (not too loose or tight)

**Red flags to watch for:**
- Cell count per image varies wildly from your expectation → check for missed cells
- Box aspect ratios very elongated → cells are circular, revisit
- Very few `non_viable` labels → confirm trypan staining is present

---

## Common Edge Cases

| Situation | Decision |
|---|---|
| Cell on a bold outer grid line | Annotate if you can identify it as a cell |
| Very small dot — could be debris or small cell | Annotate as `viable` if it's round and cell-sized; skip if too small |
| Large irregular shape | Skip — likely debris or artifact |
| Two cells that appear as a figure-8 | Two boxes: one per cell |
| Ghost/shadow of a cell | Skip — only annotate clearly in-focus cells |
| Refractile ring with no visible cytoplasm | Annotate as `viable` if it looks like a cell membrane |
