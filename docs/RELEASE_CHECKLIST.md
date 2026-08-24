# Public-release checklist

This repository was prepared from a new Git history. Complete the remaining
publication-metadata and hosting checks before the paper release tag:

- [x] Build the public release from a new Git history, excluding private
  transfer/deployment helpers and their historical objects.
- [x] Confirm that the release tree, Git objects, and package candidates contain
  no credentials, private machine paths, or authenticated remote URLs.
- [x] Confirm that `src/tbfe/preprocess/assets/Precomputed.exr` may be retained.
  The authors identified it as a redistributable placeholder EXR.
- [x] Confirm the Factory smoke crop may be publicly redistributed under the
  limited terms stated in `THIRD_PARTY_NOTICES.md`.
- [x] Add the repository URL to `README.md` and `CITATION.cff`.
- [ ] Replace the provisional BibTeX with the ACM entry and add the paper DOI
  and project page when the publication metadata becomes public.
- [x] Have the institution/project owner approve the MIT license.
- [x] Review the author list, spelling, affiliations, emails, and corresponding
  author one final time.
- [x] Add and pass `python tools/audit_release.py`; keep it as the first CI
  check for every public commit and tag.
- [x] Create the first commit from this clean repository; do not copy the old
  `.git` directory or add a remote pointing at an accidental private fork.
- [x] Enable GitHub secret scanning and push protection, and run CI before
  tagging `v1.0.0-paper`.
