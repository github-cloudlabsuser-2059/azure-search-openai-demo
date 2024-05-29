# Backend Application Documentation

The backend of the application is built using Quart, a Python framework for asynchronous web applications. The code for the backend is located in the `app/backend` directory.

## Key Components

### ADLSGen2ListFileStrategy

The `ADLSGen2ListFileStrategy` class is responsible for managing the file listing strategy when the application interacts with Azure Data Lake Storage Gen2. It provides methods to list and filter files based on their extensions. The implementation of this class can be found in `app/backend/prepdocslib/listfilestrategy.py`.

### File Processors

The `setup_file_processors` function, defined in `app/backend/prepdocs.py`, sets up file processors for different file types. These processors are responsible for parsing files and splitting text into sentences. This functionality is crucial for the application as it allows the processing of various file types like .txt, .pdf, .docx, etc.

### Chat Approach

The `system_message_chat_conversation` function, located in `app/backend/approaches/chatapproach.py`, handles the chat-based approach of the application. It processes the user's input, generates a response using the application's logic, and returns the response to the user.

## Testing

The backend application includes a suite of unit tests located in the `tests/` directory. These tests make use of mock functions such as `mock_delete_documents`, `mock_upload_documents`, and `mock_validate_token_success` to simulate the behavior of the application. These tests ensure the robustness and reliability of the application by testing its various components under different scenarios.

## Customization

For more details on customizing the backend, refer to the Customizing the backend section in the customization guide. This guide provides detailed instructions on how to modify the application's settings, add new features, and adapt the application to specific needs.
