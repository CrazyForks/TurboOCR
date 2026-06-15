# Profile image: builds on top of turboocr:vlm, recompiling only the two
# VLM source files that now carry VLM_PROFILE=1 timing instrumentation.
FROM turboocr:vlm

# Copy instrumented sources (staged in docker/ during build)
COPY docker/vlm_formula_instrumented.cpp /app/src/formula/vlm_formula.cpp
COPY docker/vlm_table_instrumented.cpp   /app/src/table/vlm_table.cpp

# Incremental rebuild: only modified TUs + relink
RUN cd /app/build && make -j$(nproc) turbo_ocr_formula turbo_ocr_table turboocr-server 2>&1 | tail -30

# Analysis script
COPY tools/analyze_vlm_profile.py /app/tools/analyze_vlm_profile.py

ENV VLM_PROFILE=1
ENV VLM_PROFILE_PATH=/tmp/vlm_profile.jsonl
