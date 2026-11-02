Copied from LSceneLLM (`utils/capeval`, itself LL3DA's caption scorers), unmodified, so XR
predictions are scored with the code LSceneLLM's validation uses. LSceneLLM's copy lacks
`meteor/data/paraphrase-en.gz`, without which the METEOR jar exits; the standard file
(identical to Chat-Scene's and LL3DA's `utils/capeval/meteor/data/paraphrase-en.gz`) is added.
