def increment_path(path, exist_ok=False, sep='', mkdir=False):
    from pathlib import Path
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(1, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path



def prepare_dataset(config):
    if config.load_from_disk is None:
        for i in range(-1, config.NFOLDS):

            train_df = df_train[df_train['fold']==i].reset_index(drop=True)

            if i==config.trn_fold:
                config.valid_stride = True
            if i!=config.trn_fold and config.downsample > 0:
                train_df = downsample_df(train_df, config.downsample)
                config.valid_stride = False

            print(len(train_df))
            ds = Dataset.from_pandas(train_df)

            ds = ds.map(
              tokenize_row,
              batched=False,
              num_proc=2,
              desc="Tokenizing",
            )

            ds.save_to_disk(f"{config.save_dir}/fold_{i}.dataset")
            with open(f"{config.save_dir}/train_pkl", "wb") as fp:
                pickle.dump(train_df, fp)
            print("Saving dataset to disk:", config.save_dir)