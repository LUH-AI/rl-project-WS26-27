# necessary steps to move to the folder on the login node
# ssh luis
# cd /bigwork/nhwpchem/Repo/rl-project
# uvenv # this will activate the virtual environment

#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH -J job_name
#SBATCH -t 0-03:00:00
#SBATCH --mail-type fail,end
#SBATCH --mail-user m.che@ai.uni-hannover.de
#SBATCH -p ai
#SBATCH --exclude=ai-n[001-004],ai-n[009-011]
#SBATCH --output experiments/luis_cluster_scripts/log/%A_task%a.out
#SBATCH --error  experiments/luis_cluster_scripts/log/%A_task%a.err