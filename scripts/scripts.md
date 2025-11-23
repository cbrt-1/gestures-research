# Scripts

| Script                 | Function                                                                                                    |
|------------------------|-------------------------------------------------------------------------------------------------------------|
| preprocess_hagrid.py   | Converts hagrid images into 138 x 8 .npy files. The script processes the hagrid zip file.                   |
| preprocess_jester.py   | Converts jester videos into 138 x video frames .npy files. The script processed the unzipped jester videos. |
| preprocess_wlasl.py    | Converts wlasl videos into 138 x video frames .npy files. The script processed the unzipped wlasl videos.   |
| print_zip_file_tree.py | Prints a zip's file tree without uncompressing it. Similar to `unzip -l`.                                   |
| realtime_test.py       | View and debug the 138 dimensional vector with a live camera.                                               |


All preprocessing scripts are resumable.