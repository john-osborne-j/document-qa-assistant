## 🐛 Troubleshooting
Common Issues

- "Ollama not found": Ensure Ollama is installed and running
- "Model not found": Run ollama pull llama3.2
- Memory issues: Reduce chunk size or use a smaller model
- Slow processing: Consider using GPU acceleration with Ollama

## Debug Mode
- Enable debug logging by adding:
```
pythonimport logging
logging.basicConfig(level=logging.DEBUG)
```
