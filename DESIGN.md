# Design System

## Direction

Simple Backtest presents itself like a clear research instrument used in daylight: precise plots, explicit assumptions, and no trading-floor theater. The public site uses a committed cobalt field against pure white, with saffron reserved for meaningful signals such as trade markers.

## Color

All authored colors use OKLCH.

| Role | Token | Value | Use |
| --- | --- | --- | --- |
| Background | `--color-bg` | `oklch(1 0 0)` | Main page field |
| Surface | `--color-surface` | `oklch(0.96 0.01 262)` | Quiet code and data regions |
| Ink | `--color-ink` | `oklch(0.18 0.025 262)` | Primary copy |
| Muted | `--color-muted` | `oklch(0.46 0.03 262)` | Secondary copy |
| Primary | `--color-primary` | `oklch(0.34 0.159 262.4)` | Hero field, links, actions, chart line |
| Primary dark | `--color-primary-dark` | `oklch(0.27 0.14 262.4)` | Active states |
| Signal | `--color-signal` | `oklch(0.86 0.15 83)` | Trade markers and decision highlights |
| Rule | `--color-rule` | `oklch(0.86 0.012 262)` | Dividers and chart guides |

White text is used on cobalt. Ink text is used on white, frost, and saffron.

## Typography

- Display and prose: Familjen Grotesk, loaded from Google Fonts with a compact sans-serif fallback stack.
- Code and command text: JetBrains Mono, loaded alongside the display family with a monospace fallback.
- Display headings use fluid sizing, balanced wrapping, and tracking no tighter than `-0.03em`.
- Body copy stays between 60 and 70 characters per line where practical.

## Layout

- Content width: `80rem` maximum with fluid page gutters.
- Spacing follows a 4-point-derived scale: 4, 8, 12, 16, 24, 32, 48, 64, and 96 pixels.
- The hero is intentionally asymmetric: message on a cobalt field, working artifact on white.
- Sections alternate between generous narrative space and denser technical ledgers.
- Grouping relies on alignment, rules, and whitespace rather than repeated cards.

## Components

- Brand mark: a stepped equity line rendered as inline SVG.
- Actions: solid cobalt or white-on-cobalt treatments with visible focus rings and 44-pixel minimum targets.
- Code surfaces: restrained frost background, thin rules, modest radius, and horizontal overflow at narrow widths.
- Scope ledger: paired supported/out-of-scope columns that collapse into a readable vertical sequence.
- Chart: semantic inline SVG with saffron trade markers and an accessible text alternative.

## Motion

- One first-load sequence reveals the cobalt field, copy, and research artifact.
- The equity curve traces once from left to right.
- Hover and press feedback stays within 100–250 milliseconds.
- `prefers-reduced-motion: reduce` disables choreography and smooth scrolling.

## Responsive Behavior

- Mobile-first layout with explicit checks at 320, 768, 1024, and 1440 pixels.
- The hero stacks at tablet widths; its message remains first in source order.
- Nonessential navigation links hide on narrow screens while installation and GitHub remain reachable in the page.
- Code scrolls rather than shrinking below a readable size.
- All touch targets remain at least 44 by 44 pixels.

## Accessibility

- WCAG 2.1 AA is the minimum target.
- Every interactive element is keyboard reachable and has a visible `:focus-visible` state.
- The chart has a title and description; color is not its only explanatory signal.
- The copy-control confirmation is announced through a polite live region.
