# GGUF W0 entrance: honest, bounded, read-only byte provenance

**Status:** SOURCE_ONLY parser and fingerprint; 5 tiny local-format tests.
No Lappy/VN1/VN2/NAS file was read or modified. Neither a filename nor an
LLM's explanation establishes what model or identity the bytes implement.

The Labyrinth already has `pretrained_model_provenance.py` for canonical
Hugging Face **safetensors** inventories, and `entrance_state_v1.py` can hash
GGUF in a mixed directory while expecting separate config/tokenizer files.
Neither specifically verifies the GGUF binary structure or distinguishes
its quantized **inference** identity from a separately trainable HF genome.

`labyrinth.gguf_entrance_fingerprint_v1` closes this narrow gap:

- Opens one explicit local absolute `.gguf` file (and optionally one separate
  GGUF LoRA adapter) in **read-only mode**. Refuses symbolic/junction paths.
- Parses little-endian GGUF **v3** header, metadata entries, tensor names,
  dimensions, types and data offsets with 64 MiB parser budget, bounded
  strings, arrays, entry counts, tensor count and alignment checks. Rejects
  malformed, oversized and ambiguous structures without deserializing weight
  tensors. Big-endian GGUF, structural versions other than v3 and deeply
  nested arrays are not supported by this pilot.
- Streams **every file byte** into SHA-256 with 1 MiB chunks; checks regular
  file identity and size before/after, without loading a 3 GiB model into RAM.
- Records numeric `general.file_type` and physical GGML tensor type counts
  as *metadata observations*, not an inferred training procedure. For
  example, `general.file_type=16` is conventionally Q5_K_S. It does not mean
  every tensor has the same quantization, or that the model can be trained.
- For a second GGUF adapter, inspects declarations `general.type=adapter`,
  `adapter.type=lora` and `general.architecture`; distinguishes declared
  match, mismatch, missing architecture or unverified adapter format.
  **Matching architecture alone cannot establish that the adapter is
  compatible with or applied to the base**. An adapter sharing a claimed
  architecture may still have incompatible tensor names/dimensions.
- Returns a deterministic in-memory receipt and manifest SHA-256; reopening
  with `verify_gguf_entrance` requires a separately supplied trusted pin,
  then rehashes physical model and adapter. There is no automatic witness,
  write, model execution or promotion.

The boolean fields `gguf_model_loaded`, `adapter_applied_proven`,
`hf_trainable_genotype_proven`, `model_identity_proven`, `origin_attested`,
`neural_improvement_proven` and `graduation_authorized` are always `false`.

## Operational implications for Vera and Labyrinth

Patrick identifies the authoritative Windows model area as `D:\VERA`, with
`C:\VERA` obsolete. The intended `model_q5_k_s.gguf` is a runtime candidate,
not proof of a valid trainable full-state W0. File names, historical handoffs
and obsolete absolute paths must never substitute for live byte inspection.
This source-only change does **not** inspect or hash that actual private file.

The two legitimate, currently **unproven** tracks are distinct:

1. **Runtime W0 (GGUF):** physical single-file fingerprint, optional GGUF
   LoRA fingerprint, actual KoboldCPP/llama.cpp load and public dialogue probes.
2. **Trainable W0 (HF):** full safetensors + tokenizer/config + optimizer,
   scheduler, RNG and cursor, captured by `campaign_full_state_v1`, and edited
   only under a physically controlled TRAIN step or bounded weight-space
   candidate path.

If the two represent related parameters, that relationship needs an explicit
conversion/quantization/adapter provenance receipt containing both physical
hashes, transformation version, parameters, and reproducible round-trip
or mapped-output tests. **Equal names or claimed lineage cannot prove
weight equivalence**. A GGUF fingerprint cannot be passed off as an HF
restorable optimizer checkpoint.

Test: `python -m unittest -q tests.test_gguf_entrance_fingerprint_v1`

Primary format sources:

- [ggml's GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [ggml's GGUF header declarations](https://github.com/ggml-org/ggml/blob/master/include/gguf.h)
- [llama.cpp's LoRA adapter loader](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-adapter.cpp)
- [llama.cpp's LoRA conversion](https://github.com/ggml-org/llama.cpp/blob/master/convert_lora_to_gguf.py)
