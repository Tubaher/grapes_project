
output_dir="stuff/dataset/concat_dataset"

mkdir -p $output_dir
mkdir -p $output_dir/data
mkdir -p $output_dir/train
mkdir -p $output_dir/val
touch $output_dir/base_names.txt

for var in "$@"
do
    echo "Copying files from $var to $output_dir"
    cp ${var}data/* $output_dir/data/
    cp ${var}train/* $output_dir/train/
    cp ${var}val/* $output_dir/val/
    echo `cat "$var"base_names.txt >> $output_dir/base_names.txt` 
done


