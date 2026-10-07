# EasyBank metadata

This repository never includes construction samples, image-derived EasyBank tensors, or other weights. Build schema-v2 banks from an authorized construction split that is disjoint from evaluation and has pixel-level lesion masks by using `code/launch_scripts/build_subtle_lesion_banks_medgemma15.sh`.

`subtle_lesion_medgemma15/MANIFEST.json` and `SHA256SUMS` identify legacy schema-v1 artifacts used by the original runs. The runtime can read them for compatibility, but labels their output `legacy_approximation` because their construction predates the equation-literal schema-v2 builder.

The source package does not publish a schema-v2 bank manifest because the private construction samples and resulting tensors are outside the release boundary. Do not treat locally reconstructed banks as the artifacts behind the paper's reported results.
