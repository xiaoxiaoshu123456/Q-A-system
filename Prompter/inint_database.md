1:初始化文档

- 支持上传 TXT 和 Markdown,PDF、Word 格式的文档等一下文档都要支持.

- ```
  # 1：定义支持的文档类型（文件扩展名）
  document_loads = {
      # 文本文件使用 TextLoader
      ".txt": TextLoader,
      # PDF文件使用 OCRPDFLoader
      ".pdf": OCRPDFLoader,
      # 文档文件使用 OCRDOCLoader
      ".doc": OCRDOCLoader,
      ".docx": OCRDOCLoader,
      # PPT文件使用 OCRPPTLoader
      ".ppt": OCRPPTLoader,
      ".pptx": OCRPPTLoader,
      # 图片文件使用 OCRIMGLoader
      ".jpg": OCRIMGLoader,
      ".png": OCRIMGLoader,
      ".jpeg": OCRIMGLoader,
      # md 文件使用 UnstructuredMarkdownLoader
      ".md": UnstructuredMarkdownLoader,
  }
  
  ```

- 把C:\Users\dujiaojiao\Desktop\q_w_System\人工智能就业课课程大纲.md

  和C:\Users\dujiaojiao\Desktop\q_w_System\人工智能就业课课程大纲.txt

- 通过url:C:\Users\dujiaojiao\Desktop\q_w_System\model\bge-m3 进行分块解析

- 按照以下分块

- ```
  # 检索参数配置
  [retrieval]
  parent_chunk_size = 1200
  child_chunk_size = 300
  chunk_overlap = 50
  retrieval_k = 5
  candidate_m = 2
  ```

  保存到milvus向量数据库中,地址如下

  ```
  # Milvus 配置
  [milvus]
  host = 192.168.88.101
  port = 19530
  # 记得创建这个表
  database_name = xxs
  collection_name = edurag_sh01
  ```
  
  要求:milvus向量数据库
  
  ```
  链接地址:
  [milvus]
  host = 192.168.88.101
  port = 19530
  # 记得创建这个表
  database_name = xxs
  collection_name = edurag_sh01
  
  mivus向量数据库中的
  注意:每个文档的grandpa_id是一样的,例如把C:\Users\dujiaojiao\Desktop\q_w_System\人工智能就业课课程大纲.md是一个grandpa_id
  id:md5的id  text:字块文本数据  dense_vector:子块稠密向量 sparse_vector:稀疏向量  parend_id:父块id  parend_context:父块内容
  ```
  
  检查milvus中是否有数据
  
  