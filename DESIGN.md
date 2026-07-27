# Design System

## Direction

Simple Backtest presents itself as a research instrument read in daylight: ruled paper, measured
type, and a single red pen. The public site is drawn as a schematic of the instrument rather than
as a pitch for it — what goes in, what the engine does, what comes out, and, on a hard accent rule
beneath all of it, what the instrument refuses to do. Scope is a structural element of the page,
not a disclaimer appended to it.

The canonical tokens live in [`website/tokens.css`](website/tokens.css). Every colour and font in
`website/styles.css` references one by name; no stylesheet declares a raw colour or font stack.

## Color

All authored colors use OKLCH. Neutrals are tinted warm toward the accent hue; neither paper nor
ink is a pure extreme.

| Role | Token | Value | Use |
| --- | --- | --- | --- |
| Paper | `--color-paper` | `oklch(0.978 0.006 75)` | Page field |
| Paper 2 | `--color-paper-2` | `oklch(0.955 0.008 72)` | Command and code surfaces |
| Paper 3 | `--color-paper-3` | `oklch(0.922 0.01 72)` | Reserved third elevation |
| Ink | `--color-ink` | `oklch(0.215 0.014 55)` | Primary copy, heavy rules |
| Ink 2 | `--color-ink-2` | `oklch(0.395 0.013 55)` | Secondary copy, schematic lists |
| Muted | `--color-muted` | `oklch(0.53 0.012 58)` | Captions, notes, control boundaries |
| Rule | `--color-rule` | `oklch(0.845 0.01 68)` | Hairline dividers |
| Rule 2 | `--color-rule-2` | `oklch(0.912 0.008 70)` | Chart grid |
| Accent | `--color-accent` | `oklch(0.505 0.185 32)` | The pen: one hero line, the scope rule, links, the trace |
| Accent ink | `--color-accent-ink` | `oklch(0.985 0.006 75)` | Text on any future accent fill (verified 6.05:1) |
| Focus | `--color-focus` | `oklch(0.5 0.19 32)` | `:focus-visible` rings only |

Contrast is verified numerically, not by eye. Ink on paper 16.5:1, ink-2 on paper 8.9:1, muted on
paper 4.97:1, muted on paper-2 4.64:1, accent on paper 6.05:1, focus on paper 6.19:1. Every value
sits in the sRGB gamut.

The accent is a signal, not a surface. It occupies well under 5% of any viewport and is never used
as a large fill. Color is never the only carrier of meaning — the scope band is labelled in words.

## Typography

Two families. The display voice and the code voice are the same face, which is the point: the page
speaks in the language of the tool.

- Display and code: JetBrains Mono, self-hosted, variable 400–800. Headings set at 700.
- Prose and UI: Familjen Grotesk, self-hosted, variable 400–700. Body set at 400.
- Both are OFL-licensed, subset to latin, and served as local `woff2` with `font-display: swap`.
  No third-party font CDN.
- Scale is a 1.25 major third from a 16px body. Display clamps to a 3.25rem ceiling; the hero
  headline is three short lines, none longer than 15 characters.
- Headings are always roman. Tracking is tight on display, loose on uppercase labels.
- Tabular figures on the ledger and the notebook index.

## Layout

- Content width: `78rem` maximum with a fluid `--page-gutter`.
- Spacing is a 4-point-derived scale, nine steps, named by role.
- The hero is left-biased and bottom-heavy; the schematic spans the full content width beneath it.
- The accounting ledger is held to `62rem` so it reads as a block rather than a spread.
- Grouping relies on rules and alignment. There are no cards and no card nesting.

## Components

- Brand mark: a stepped equity line as inline SVG, stroked in accent.
- Navigation: wordmark hard-left, one link hard-right, nothing between. The space is the design.
- Schematic: a CSS-grid flow of three stages that reflows to a vertical stack with rotated
  connectors below `60rem`. Not a fixed SVG.
- Scope band: a `2px` accent rule with a labelled list of what the accounting model excludes.
- Ledger: a three-column spec table, hairline-ruled per row, collapsing to two columns and then to
  a stacked block. Table semantics are preserved with explicit ARIA roles because the rows are laid
  out with `display: grid`.
- Code surface: a typographic frame — label row, hairline, `<pre>` — never a drawn window chrome.
- Command: the install line and its copy control; the control drops to its own full-width row
  below `26rem` so the command is never clipped.
- Footer: a monospace colophon that closes the page as a record, not a sitemap.

## Motion

Three primitives, no more.

- One orchestrated first-load reveal, staggered by DOM index, capped under 500ms.
- The equity trace draws once, left to right.
- A one-pixel press on controls.

Only `transform`, `opacity`, and `stroke-dashoffset` are animated. Easings are the three named
tokens; the browser default is never used. `prefers-reduced-motion: reduce` removes the trace
animation and the reveal offset.

## Responsive Behavior

- Mobile-first; `min-width` queries only. Breakpoints at 26, 40, 60, and 90 rem.
- Verified at 320, 375, 414, 768, 1280, and 1920 px: no horizontal scroll, nothing overflowing the
  viewport, no clickable label wrapping to two lines.
- `overflow-x: clip` on both `html` and `body` — never `hidden`, which would break sticky and trap
  focus.
- Touch targets are at least 44px on coarse pointers.
- Code and commands scroll rather than shrinking below a readable size.

## Accessibility

- WCAG 2.1 AA is the minimum target, verified by computation for every text pair.
- Every interactive element ships default, hover, focus-visible, active, and disabled styling.
- Focus rings appear instantly and are never transitioned.
- The chart has a title and a description that names the values as drawn rather than measured.
- Decorative SVG and glyphs are `aria-hidden`; the copy control reports through a polite live
  region.

## Notes

`tokens.css` is the only export this project needs — the site is vanilla HTML with no Tailwind,
shadcn, or DTCG consumer. Add other formats when something actually consumes them.
