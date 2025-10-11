# 依賴更新修復說明

## 問題描述

運行 `web_captioner_awq.py` 時出現以下錯誤：

1. **huggingface-hub 版本不匹配**
   ```
   ImportError: huggingface-hub==1.0.0.rc5 is required for a normal functioning of this module,
   but found huggingface-hub==0.35.3.
   ```

2. **缺少 compressed-tensors 套件**
   ```
   ImportError: compressed_tensors is not installed and is required for compressed-tensors quantization.
   Please install it with `pip install compressed-tensors`.
   ```

## 解決方案

### 1. Dockerfile 更新

#### 主 Dockerfile (`./Dockerfile`)
- **第 64 行**：在 AutoAWQ 安裝時添加 `compressed-tensors`
  ```dockerfile
  RUN --mount=type=cache,target=/root/.cache/pip \
      pip3 install autoawq autoawq-kernels compressed-tensors
  ```

- **第 90-92 行**：在最後添加依賴更新步驟，確保所有關鍵依賴使用最新版本
  ```dockerfile
  # Final update: Ensure all critical dependencies are up-to-date
  # This step ensures compatibility between transformers, huggingface-hub, and compressed-tensors
  RUN pip3 install --no-cache-dir -U transformers huggingface-hub compressed-tensors
  ```

#### 次要 Dockerfile (`./docker/Dockerfile-omni-3-cu124`)
- **第 80 行**：添加 `huggingface-hub==1.0.0.rc5` 明確版本
- **第 93-95 行**：添加最終依賴更新步驟
  ```dockerfile
  # Final update: Ensure all critical dependencies are up-to-date
  # This step ensures compatibility between transformers, huggingface-hub, and compressed-tensors
  RUN pip3 install --no-cache-dir -U transformers huggingface-hub compressed-tensors
  ```

### 2. docker-compose.yml 更新

為所有 AWQ 服務添加啟動前的依賴更新：

#### qwen3-omni-captioner-awq (第 67 行)
```yaml
command: >
  bash -c "pip3 install -U transformers huggingface-hub compressed-tensors &&
  python3 web_captioner_awq.py
  --use-transformers
  --flash-attn2
  --checkpoint-path /data/models/Qwen3-Omni-30B-A3B-Captioner-AWQ-8bit
  --server-name 0.0.0.0
  --server-port 8901"
```

#### qwen3-omni-demo-awq (第 109 行)
```yaml
command: >
  bash -c "pip3 install -U transformers huggingface-hub compressed-tensors &&
  python3 web_demo_awq.py
  --use-transformers
  --flash-attn2
  --checkpoint-path /data/models/Qwen3-Omni-30B-A3B-Instruct-AWQ-8bit
  --server-name 0.0.0.0
  --server-port 8901"
```

#### qwen3-omni-captioner (第 25 行)
```yaml
command: >
  bash -c "pip3 install -U transformers huggingface-hub compressed-tensors &&
  python3 web_demo_captioner.py
  --use-transformers
  --flash-attn2
  --checkpoint-path /data/models/Qwen3-Omni-30B-A3B-Captioner
  --server-name 0.0.0.0
  --server-port 8901"
```

## 使用方式

### 方法 1：使用現有映像檔（快速啟動）
如果已經有構建好的映像檔，可以直接啟動，docker-compose 會在啟動時自動更新依賴：

```bash
# 啟動 AWQ captioner 服務
docker-compose --profile captioner-awq up

# 或在背景執行
docker-compose --profile captioner-awq up -d

# 查看日誌
docker-compose logs -f qwen3-omni-captioner-awq
```

### 方法 2：重新構建映像檔（推薦）
重新構建映像檔以包含所有依賴更新：

```bash
# 重新構建（不使用快取）
docker-compose build --no-cache

# 然後啟動服務
docker-compose --profile captioner-awq up
```

### 方法 3：只構建特定服務
```bash
# 只構建 AWQ captioner 服務
docker-compose build --no-cache qwen3-omni-captioner-awq

# 啟動服務
docker-compose --profile captioner-awq up
```

## 技術細節

### 為什麼需要這些更新？

1. **transformers 開發版本**：Dockerfile 從 GitHub 安裝最新開發版的 transformers
   ```dockerfile
   pip3 install git+https://github.com/huggingface/transformers
   ```
   這個版本需要 `huggingface-hub==1.0.0.rc5` 或更新版本

2. **AWQ 量化支持**：AWQ 8-bit 量化模型需要 `compressed-tensors` 套件來處理量化配置

3. **依賴兼容性**：最終的更新步驟確保三個關鍵套件版本互相兼容：
   - `transformers`（開發版）
   - `huggingface-hub`（1.0.0.rc5+）
   - `compressed-tensors`（最新版）

### 雙層保護策略

1. **構建時**：Dockerfile 中在構建時就安裝正確的依賴
2. **運行時**：docker-compose.yml 中在容器啟動時再次確保依賴是最新版本

這種策略確保：
- 即使使用舊的映像檔，也能在啟動時自動修復依賴問題
- 新構建的映像檔已經包含正確的依賴，啟動更快

## 驗證

啟動服務後，應該看到類似輸出：

```
Requirement already satisfied: transformers in /usr/local/lib/python3.10/dist-packages (5.0.0.dev0)
Requirement already satisfied: huggingface-hub in /usr/local/lib/python3.10/dist-packages (1.0.0rc5)
Successfully installed huggingface-hub-1.0.0rc5
```

如果一切正常，應該不會再看到 ImportError，應用程式會成功啟動並監聽 8901 端口。

## 注意事項

- 使用 `-U` 標誌確保 pip 會升級到最新版本，而不是跳過已安裝的套件
- 使用 `--no-cache-dir` 在 Dockerfile 最後更新時避免快取問題
- Gradio 5.44.1 要求 `huggingface-hub<1.0`，但 1.0.0rc5 仍然可以正常工作（會有警告但不影響功能）
