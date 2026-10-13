set -e
cd /home/sya225/opd/scripts
grep -q 'MAX_TOKENS' eval_v5.sh || sed -i 's|--n 4 --max_tokens 8192"|--n 4 --max_tokens ${MAX_TOKENS:-8192}"|' eval_v5.sh
grep -q 'max_tokens:' post_run.sh || sed -i 's|sed -e "s#^model:.*#model: $HF_OUT#"|sed -e "s#^model:.*#model: $HF_OUT#" -e "s#^max_tokens:.*#max_tokens: ${MAX_TOKENS:-8192}#"|' post_run.sh
grep -n 'MAX_TOKENS' eval_v5.sh post_run.sh
