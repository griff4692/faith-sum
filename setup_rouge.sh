echo $ROUGE_HOME
mkdir $ROUGE_HOME
curl -L https://github.com/Yale-LILY/SummEval/tarball/master -o project.tar.gz -s
tar -xzf project.tar.gz
mv Yale-LILY-SummEval-9b58833/evaluation/summ_eval/ROUGE-1.5.5/ $ROUGE_HOME
rm project.tar.gz
rm -rf Yale-LILY-SummEval-9b58833/
