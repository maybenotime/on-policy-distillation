#!/bin/bash
# FSDP checkpoint -> merged HF copy pairs on node 05. DRY=1 lists only.
cd /localscratch/sya225/checkpoints || exit 1
pairs="
opd_ds15b_fromDeepScaleR_dapo_16k/global_step_50:hf_DeepScaleR_step50
opd_ds15b_fromDeepScaleR_dapo_16k/global_step_200:hf_DeepScaleR_step200
opd_ds15b_fromLight7B_dapo_16k/global_step_50:hf_Light7B_step50
opd_ds15b_fromLight7B_dapo_16k/global_step_200:hf_Light7B_step200
opd_ds15b_fromNemoBroRL_dapo_16k/global_step_50:hf_NemoBroRL_step50
opd_ds15b_fromNemoBroRL_dapo_16k/global_step_200:hf_NemoBroRL_step200
opd_ds15b_tpJustRL_shapeStu/global_step_150:hf_ds15b_tpJustRL_shapeStu_step150
opd_ds15b_tpJustRL_shapeStu/global_step_200:hf_ds15b_tpJustRL_shapeStu_step200
opd_ds15b_tpR17B_shapeJustRL/global_step_50:hf_dsTP_R17B_shapeJustRL_step50
opd_ds15b_tpStu_shapeJustRL/global_step_50:hf_dsTP_Stu_shapeJustRL_step50
opd_q3i17_from4B_dapo_16k/global_step_50:hf_q3i17_from4B_step50
opd_q3i17_from4B_dapo_16k/global_step_200:hf_q3i17_from4B_step200
opd_q3i17_fromK4Bcode1200_dapo_16k/global_step_50:hf_K4Bcode1200_step50
opd_q3i17_fromK4Bcode1200_dapo_16k/global_step_200:hf_K4Bcode1200_step200
opd_q3i17_fromQ8Bdapo_dapo_16k/global_step_50:hf_Q8Bdapo_step50
opd_q3i17_fromQ8Bdapo_dapo_16k/global_step_200:hf_Q8Bdapo_step200
opd_q3i17_fromQ8Bgrpomath_dapo_16k/global_step_50:hf_Q8Bgrpomath_step50
opd_q3i17_fromQ8Bgrpomath_dapo_16k/global_step_150:hf_Q8Bgrpomath_step150
opd_q3i17_fromQ8Bgrpomath_dapo_16k/global_step_200:hf_Q8Bgrpomath_step200
opd_code_17b/global_step_424:hf_opd17b_step424
opd_code_17b_from14b/global_step_424:hf_opd17b_from14b_step424
opd_code_17b_from4b/global_step_424:hf_opd_1.7b_from4b_step424
opd_code_17b_fromK4Bcode1200/global_step_424:hf_opd_1.7b_fromK4Bcode1200_step424
opd_code_30k/global_step_424:hf_opd30k_step424
"
tot=0
for p in $pairs; do f=${p%%:*}; h=${p##*:}
  if [ ! -d "$f" ]; then continue; fi
  if [ -d "$h" ] && [ "$(find $h -maxdepth 1 -name '*.safetensors' | wc -l)" -ge 1 ] && [ "$(du -sm $h | cut -f1)" -ge 2500 ] && [ -f "$h/config.json" ]; then
    sz=$(du -sm "$f" | cut -f1); tot=$((tot+sz)); echo "OK   $f (${sz}M) <- $h"
    [ "${DRY:-1}" = 0 ] && rm -rf "$f"
  else echo "SKIP $f : hf copy $h missing/incomplete"; fi
done
echo "freeable_total=${tot}M"; ls -d hf_* | wc -l; ls -d hf_*
