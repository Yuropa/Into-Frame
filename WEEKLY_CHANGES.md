# Weekly Change Tracking

## Current Baseline

| Field | Value |
|-------|-------|
| Commit | `08ac79ab87b2d85a25c5f482e8a1d5c416b443f2` |
| Message | Archive the scene |
| Date set | 2026-06-29 |

---

## How to Get a Summary

Ask Claude Code:
> "Summarize all changes since the baseline in WEEKLY_CHANGES.md"

Or manually:
```sh
git log 08ac79ab87b2d85a25c5f482e8a1d5c416b443f2..HEAD --oneline
git diff 08ac79ab87b2d85a25c5f482e8a1d5c416b443f2..HEAD --stat
```

After reviewing, update the baseline to current HEAD:
```sh
git rev-parse HEAD  # copy this hash into the Commit field above
```

---

## Change Log

### Week ending 2026-06-29 (baseline set)

*Baseline established at `08ac79ab` — "Archive the scene" (2026-06-10).*
