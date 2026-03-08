# How to Contribute

Please read these guidelines before contributing to MediAssist:

 - [Question or Problem?](#question)
 - [Issues and Bugs](#issue)
 - [Feature Requests](#feature)
 - [Adding New Modules](#modules)
 - [Submitting a Pull Request](#pullrequest)
 - [Contributor License Agreement](#cla)


## <a name="question"></a> Got a Question or Problem?

If you have questions about how to use or extend MediAssist, please read the
[project README][readme] first. For deeper questions about the RAG pipeline,
fine-tuning setup, or deployment, open a [Discussion][discussions] on GitHub.

GitHub issues are only for [reporting bugs](#issue) and [feature requests](#feature), not
general questions or help.


## <a name="issue"></a> Found an Issue?

If you find a bug in the pipeline, a retrieval quality problem, or incorrect triage
behaviour, you can help by submitting an issue to the [GitHub Repository][github].
Even better, submit a Pull Request with a fix.

When submitting an issue please include the following information:

- A clear description of the issue
- The patient input text or file that triggered the problem
- The pipeline stage where the issue occurs (NER, retrieval, reranking, triage, generation)
- The full error message and stacktrace if an exception was thrown
- The triage level returned vs. what was expected
- Your Python version, OS, and whether you are running on Apple Silicon or x86
- If possible, include a minimal code snippet that reproduces the issue

The more detail you provide, the faster the issue can be resolved.


## <a name="feature"></a> Want a Feature?

You can request a new feature by submitting an issue to the [GitHub Repository][github].
Before requesting a feature, please consider the following:

- MediAssist is structured in numbered stages (`1_data`, `2_finetune`, `3_rag`, `4_app`,
  `5_database`, `6_ui`, `7_deploy`). Consider whether your feature fits cleanly into an
  existing stage or warrants a new one
- The RAG pipeline uses isolated ChromaDB collections — new knowledge domains should follow
  the same pattern and must not interfere with existing collections
- Features that affect triage classification or emergency detection require extra scrutiny.
  Patient safety behaviour must never regress
- This is a medical assistant — features that could produce harmful, misleading, or
  overconfident clinical output are unlikely to be accepted without strong safeguards


## <a name="modules"></a> Adding New Modules or Knowledge Updates

MediAssist is designed to be extended. The most common contributions are new RAG knowledge
collections, new data sources, and new pipeline stages. Please follow these guidelines:

### Adding a New RAG Knowledge Collection

- Create a dedicated indexing script: `3_rag/index_<domain>.py`
- Target a **new, uniquely named ChromaDB collection** — never write to `medical_knowledge`
  or any existing collection from a new script
- Tag every chunk with source metadata: `{"collection": "<name>", "source": "<origin>", "category": "<type>"}`
- Include a `--rebuild` flag that wipes and rebuilds only your collection
- The `MedicalRetriever` in `pipeline.py` auto-discovers non-empty collections —
  no changes to the retriever are needed for new collections
- Verify that running your script does not change the chunk count of existing collections

### Adding a New Pipeline Stage

- Add new stages as self-contained classes in `3_rag/pipeline.py` or as separate files
  in the appropriate numbered directory
- Do not modify existing class interfaces (`MedicalNER`, `MedicalRetriever`, `Reranker`,
  `MediAssistPipeline`) in breaking ways
- New stages must degrade gracefully — if a dependency is missing, the pipeline should
  fall back to existing behaviour rather than crash

### Updating the Fine-Tuned Model

- Data preparation changes go in `1_data/prepare_data.py`
- Training configuration changes go in `2_finetune/config.yaml`
- Always run with a held-out validation set and report final validation loss
- Push new model versions to HuggingFace with a clear commit message describing
  what changed (dataset, hyperparameters, base model version)
- Update `HF_MODEL_REPO` in `.env` and document the change

### Updating the UI

- The Streamlit UI (`6_ui/app.py`) uses a dark clinical theme with strict CSS variables.
  New UI elements must use existing CSS variables (`--bg`, `--accent`, `--border`, etc.)
  and match the existing visual language
- New sidebar sections must use `_sidebar_section(title)` for consistency
- Emergency and triage display behaviour must not be altered without strong justification


## <a name="pullrequest"></a> Submitting a Pull Request

When submitting a pull request to the [GitHub Repository][github], please do the following:

- Check that your code follows the existing style and naming conventions used across
  the numbered pipeline stages
- Confirm that the existing ChromaDB collections are unaffected by your changes —
  run `python 3_rag/index.py` and verify the chunk count matches before and after
- If you modified `pipeline.py`, test end-to-end with at least one LOW, one MEDIUM,
  and one HIGH triage query to confirm behaviour is intact
- If you added a new RAG collection, include sample queries that demonstrate retrieval
  from your collection working correctly
- Document any new environment variables required in `.env.example`
- Do not commit `.env`, `chroma_db/`, `adapters/`, or `fused_model/` directories —
  these are gitignored for good reason

Read [GitHub Help][pullrequesthelp] for more details about creating pull requests.


## <a name="cla"></a> Contributor License Agreement

By contributing your code to MediAssist you grant Kishan Murali a non-exclusive,
irrevocable, worldwide, royalty-free, sublicenseable, transferable license under all
of your relevant intellectual property rights (including copyright, patent, and any
other rights), to use, copy, prepare derivative works of, distribute and publicly
perform and display the contributions on any licensing terms, including without
limitation: (a) open source licenses like the MIT license; and (b) binary, proprietary,
or commercial licenses. Except for the licenses granted herein, you reserve all right,
title, and interest in and to the contribution.

You confirm that you are able to grant these rights. You represent that you are legally
entitled to grant the above license. If your employer has rights to intellectual property
that you create, you represent that you have received permission to make contributions
on behalf of that employer, or that your employer has waived such rights for the
contributions.

You represent that the contributions are your original works of authorship, and to your
knowledge, no other person claims, or has the right to claim, any right in any invention
or patent related to the contributions. You also represent that you are not legally
obligated, whether by entering into an agreement or otherwise, in any way that conflicts
with the terms of this license.

Kishan Murali acknowledges that, except as explicitly described in this Agreement, any
contribution which you provide is on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES
OR CONDITIONS OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR
PURPOSE.


[github]: https://github.com/kisht2t/mediassist
[readme]: https://github.com/kisht2t/mediassist/blob/main/README.md
[discussions]: https://github.com/kisht2t/mediassist/discussions
[pullrequesthelp]: https://help.github.com/articles/using-pull-requests
