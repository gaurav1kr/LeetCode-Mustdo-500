
# System Overview: Google Drive

Google Drive is a cloud-based file storage and synchronization service developed by Google. It allows users to store and share files online and synchronize files across multiple devices. It is tightly integrated with other Google services like Google Docs, Sheets, and Slides.

## High-Level Design (HLD)

### Key Components of Google Drive:
- **Storage System**: Google Drive stores files on distributed storage systems.
- **Metadata Management**: Metadata related to files (e.g., name, size, owner) is handled by a database.
- **File Synchronization**: Google Drive synchronizes files across devices using algorithms to detect changes and efficiently sync files.
- **Collaboration**: Google Drive enables real-time collaboration on documents.
- **APIs**: Google Drive provides APIs for file management and storage.

## Low-Level Design (LLD)

### Detailed Components
- **File Storage**: Divided into block storage and object storage for scalability.
- **Metadata Storage**: Stored in databases like Bigtable or Spanner.
- **Sync Service**: Detects changes in files and synchronizes them across devices.
- **Collaboration Layer**: Allows real-time edits and updates to shared files.

## Data Flow:
1. **File Upload**: Files are uploaded from clients and stored in the cloud after being split into chunks.
2. **Sync**: Files are synchronized between devices using checksums and diff algorithms.
3. **Collaboration**: Changes to files are synced in real time using protocols like WebSocket.

## Class Diagram

```plaintext
+---------------------+
|      User           |
+---------------------+
| - userID: string    |
| - userName: string  |
+---------------------+
| + login()           |
| + logout()          |
+---------------------+
          |
          v
+---------------------+
|      File           |
+---------------------+
| - fileID: string    |
| - fileName: string  |
| - fileSize: long    |
| - owner: User*      |
+---------------------+
| + upload()          |
| + download()        |
| + share()           |
+---------------------+
          |
          v
+---------------------+
|    StorageService   |
+---------------------+
| - storageID: string |
| - storageSize: long |
+---------------------+
| + uploadFile()      |
| + downloadFile()    |
| + syncFile()        |
+---------------------+
```

## Schema Design

**User Table**:
```plaintext
UserTable
- userID (PK)
- userName
- email
- passwordHash
```

**File Table**:
```plaintext
FileTable
- fileID (PK)
- userID (FK)
- fileName
- fileSize
- fileVersion
- timestamp
```

**Permission Table**:
```plaintext
PermissionTable
- permissionID (PK)
- fileID (FK)
- userID (FK)
- accessLevel (read/write)
```

## APIs

- **Upload File**: `POST /upload`
- **Download File**: `GET /download/{fileID}`
- **Share File**: `POST /share/{fileID}`
- **List Files**: `GET /files`

## Services

- **Authentication Service**: Handles OAuth 2.0 for user authentication.
- **Sync Service**: Manages synchronization between devices.
- **Collaboration Service**: Manages real-time file collaboration.
- **Versioning Service**: Stores and retrieves file versions.

## C++ Code Example: Multithreading and File Upload

```cpp
#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <chrono>
#include <memory>

// Mutex for thread safety
std::mutex file_mutex;

class User {
public:
    std::string userID;
    std::string userName;
    std::string email;

    User(const std::string& id, const std::string& name, const std::string& email)
        : userID(id), userName(name), email(email) {}

    void login() {
        std::cout << userName << " logged in successfully." << std::endl;
    }

    void logout() {
        std::cout << userName << " logged out." << std::endl;
    }
};

class File {
public:
    std::string fileID;
    std::string fileName;
    size_t fileSize;
    std::shared_ptr<User> owner;

    File(const std::string& id, const std::string& name, size_t size, std::shared_ptr<User> user)
        : fileID(id), fileName(name), fileSize(size), owner(user) {}

    void upload() {
        std::lock_guard<std::mutex> lock(file_mutex);
        std::cout << "Uploading file: " << fileName << ", Size: " << fileSize << " bytes" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2)); // Simulate upload delay
        std::cout << "File " << fileName << " uploaded successfully!" << std::endl;
    }

    void sync() {
        std::lock_guard<std::mutex> lock(file_mutex);
        std::cout << "Synchronizing file: " << fileName << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1)); // Simulate sync delay
        std::cout << "File " << fileName << " synchronized successfully!" << std::endl;
    }

    void share() {
        std::cout << "Sharing file: " << fileName << " with other users." << std::endl;
    }
};

class StorageService {
public:
    void uploadFile(const std::shared_ptr<File>& file) {
        std::cout << "Storing file: " << file->fileName << std::endl;
        file->upload();
    }

    void syncFile(const std::shared_ptr<File>& file) {
        std::cout << "Syncing file: " << file->fileName << std::endl;
        file->sync();
    }
};

class SyncService {
public:
    void syncFiles(const std::vector<std::shared_ptr<File>>& files) {
        for (auto& file : files) {
            std::cout << "Syncing file: " << file->fileName << std::endl;
            file->sync();
        }
    }
};

// Main function to simulate file upload and sync
int main() {
    // Create Users
    auto user1 = std::make_shared<User>("1", "Alice", "alice@example.com");
    auto user2 = std::make_shared<User>("2", "Bob", "bob@example.com");

    // Create Files
    std::shared_ptr<File> file1 = std::make_shared<File>("F1", "document1.txt", 1024, user1);
    std::shared_ptr<File> file2 = std::make_shared<File>("F2", "image2.jpg", 2048, user2);

    // Create Storage and Sync Services
    StorageService storageService;
    SyncService syncService;

    // Create Threads for Upload and Sync
    std::vector<std::thread> threads;
    threads.push_back(std::thread(&StorageService::uploadFile, &storageService, file1));
    threads.push_back(std::thread(&StorageService::uploadFile, &storageService, file2));

    for (auto& t : threads) {
        t.join();
    }

    // Now syncing files
    std::vector<std::shared_ptr<File>> filesToSync = { file1, file2 };
    syncService.syncFiles(filesToSync);

    // Logout users after task completion
    user1->logout();
    user2->logout();

    return 0;
}

```

### The diagram outlines the flow of file upload and synchronization between client, file service, storage service, and sync service.

## Conclusion

This document provides an overview of the Google Drive architecture, including high-level design, low-level design, schema, and a simple C++ code example for multithreading and file synchronization. The concepts discussed here apply to scalable cloud storage systems.
