The project can be run via 
```bash
python arxiv_processor.py
```

This will download the most recent arXiv paper from each of the categories listed in the file. Adding the 
```--no-download``` flag will process only the files already downloaded locally. The default model is Gemini 2.5 Flash, but o3 is also an option. 

Running 
```bash
python arxiv_processor.py --model gemini --no-download
```
will 

The problems and answers can be rendered in your browser using 
```bash
python render_output.py && open output.html
```
This doesn't work very well though, so it's more reliable to convert the jsons to LaTeX and manually compile in overleaf using
```bash
python export_to_tex.py                               
```