The project can be run via 
```bash
python arxiv_processor.py
```

This will download the most recent arXiv paper from each of the categories listed in the file. Adding the 
```--no-download``` flag will process only the files already downloaded locally.

The problems and answers can be rendered in your browser using 
```bash
python render_output.py && open output.html
```