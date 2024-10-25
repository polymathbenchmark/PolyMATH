# PolyMATH

## General Workflow Guidelines
> [!TIP]
> Always `git pull` to ensure you have the latest changes in the main branch.

> [!IMPORTANT]  
> Always push to your remote branch.

> [!NOTE]  
> Create PR.

## Benchmark Creation Steps
#### Step 0: 
Run annotation_utils.sh in the "create_metadata" mode. 
- metadata json (updates if it already exists) in the sub-folder for each paper
- creates the directory structure
```
python ./scripts/annotation_utils.py \
--mode create_metadata \
--paper_path "./<PAPER_NAME_WITH_EXTENSION>" \
--annotator_name kscaria \
```

#### Step 1: 
Go to <paper>/screenshots/ and add images. Naming convention:
- `q<n>_<part>_[c<context>]`
-   `[]:optional`
- `c<num>`

#### Step 2: 
Run annotation_utils.sh in the "create_annotations" mode. 
- annotation csv within the individual folder. Populates all the fields it can
- If the datastore path is given, the script will iterate through all sub-folders and create the annotations.csv for all of them.
- If `-paper_path` is given, the script will only create the annotations.csv for the provided paper path.
```
python ./scripts/annotation_utils.py \
--mode create_annotation \
--datastore_path "./datastore/"
```

#### Step 3: 
Enter the manual annotation fields.

#### Step 4: (WIP)
Run a script that converts 
- annotation csv to json with the index and subindex structure we decided. \[Annotation json gets saved to global folder.\]

[raw_dataset link][https://drive.google.com/drive/u/0/folders/1MVpNaPzsTb7Mf03QyvCwZNdM5SUlPawT]
