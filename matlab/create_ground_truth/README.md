# matlab code to re-create the DrEYEve dataset ground truth


[`main.m`](./main.m)

    Main script that loop over dataset runs and for each run start ground truth recomputation
    Output directory is defined here in `config('output_root')`
    
[`create_new_ground_truth.m`](./create_new_ground_truth.m)

    Function that takes the ID of a run as input and recompute the ground truth for every frame
    Each ground truth frame is saved in `config('output_root')/<run_id>/%06d.png`
    
[`set_lower_bound_for_each_run.m`](./set_lower_bound_for_each_run.m)

    loop over each run of the dataset and annotate the lower row (higher y) in which is possible to have GT
    when computing saliency maps, every y coordinate that is greater than the highest y allowed will be considered an error and discarded 

    

    
