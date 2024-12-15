from typing import Optional, Dict, List, Any
from literalai import LiteralClient
from datetime import datetime
import logging


class UserManagement:
    def __init__(self, literal_client: LiteralClient):
        """
        Initialize UserManagement with a LiteralClient instance

        Args:
            literal_client (LiteralClient): Initialized LiteralAI client
        """
        self.client = literal_client
        self.logger = logging.getLogger(__name__)

    def get_users(
            self,
            first: Optional[int] = None,
            after: Optional[str] = None,
            before: Optional[str] = None,
            filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve a list of users with pagination and filtering options

        Args:
            first (Optional[int]): Number of users to retrieve
            after (Optional[str]): Cursor for pagination - fetch records after this
            before (Optional[str]): Cursor for pagination - fetch records before this
            filters (Optional[Dict]): Filters to apply to the query

        Returns:
            Dict[str, Any]: Dictionary containing paginated user data

        Raises:
            Exception: If there's an error fetching users
        """
        try:
            response = self.client.api.get_users(
                first=first,
                after=after,
                before=before,
                filters=filters
            )

            self.logger.info(f"Successfully retrieved {len(response.get('edges', [])) if response else 0} users")
            return response

        except Exception as e:
            self.logger.error(f"Error fetching users: {str(e)}")
            raise

    def get_user(
            self,
            id: Optional[str] = None,
            identifier: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single user by ID or identifier

        Args:
            id (Optional[str]): The unique ID of the user
            identifier (Optional[str]): Alternative identifier (username/email)

        Returns:
            Optional[Dict[str, Any]]: User data if found, None otherwise

        Raises:
            ValueError: If neither id nor identifier is provided
        """
        if not id and not identifier:
            raise ValueError("Either id or identifier must be provided")

        try:
            user = self.client.api.get_user(id=id, identifier=identifier)
            if user:
                self.logger.info(f"Successfully retrieved user: {id or identifier}")
            else:
                self.logger.warning(f"User not found: {id or identifier}")
            return user

        except Exception as e:
            self.logger.error(f"Error fetching user {id or identifier}: {str(e)}")
            raise

    def create_user(
            self,
            identifier: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new user with the specified identifier and metadata

        Args:
            identifier (str): Unique identifier for the user (username/email)
            metadata (Optional[Dict]): Additional user data

        Returns:
            Dict[str, Any]: Created user data

        Raises:
            ValueError: If identifier is empty
        """
        if not identifier:
            raise ValueError("Identifier cannot be empty")

        try:
            # Add created_at to metadata
            metadata = metadata or {}
            metadata['created_at'] = datetime.utcnow().isoformat()

            user = self.client.api.create_user(
                identifier=identifier,
                metadata=metadata
            )

            self.logger.info(f"Successfully created user: {identifier}")
            return user

        except Exception as e:
            self.logger.error(f"Error creating user {identifier}: {str(e)}")
            raise

    def update_user(
            self,
            id: str,
            identifier: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Update an existing user's information

        Args:
            id (str): The unique ID of the user to update
            identifier (Optional[str]): New identifier for the user
            metadata (Optional[Dict]): New or updated metadata

        Returns:
            Dict[str, Any]: Updated user data

        Raises:
            ValueError: If id is empty
        """
        if not id:
            raise ValueError("User ID cannot be empty")

        try:
            # Add updated_at to metadata
            metadata = metadata or {}
            metadata['updated_at'] = datetime.utcnow().isoformat()

            user = self.client.api.update_user(
                id=id,
                identifier=identifier,
                metadata=metadata
            )

            self.logger.info(f"Successfully updated user: {id}")
            return user

        except Exception as e:
            self.logger.error(f"Error updating user {id}: {str(e)}")
            raise

    def delete_user(self, id: str) -> Dict[str, Any]:
        """
        Delete a user by their ID

        Args:
            id (str): The unique ID of the user to delete

        Returns:
            Dict[str, Any]: Result of the deletion operation

        Raises:
            ValueError: If id is empty
        """
        if not id:
            raise ValueError("User ID cannot be empty")

        try:
            result = self.client.api.delete_user(id=id)
            self.logger.info(f"Successfully deleted user: {id}")
            return result

        except Exception as e:
            self.logger.error(f"Error deleting user {id}: {str(e)}")
            raise

    def get_or_create_user(
            self,
            identifier: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get a user by identifier or create if they don't exist

        Args:
            identifier (str): Unique identifier for the user
            metadata (Optional[Dict]): Metadata for new user if created

        Returns:
            Dict[str, Any]: User data (either existing or newly created)
        """
        try:
            # First try to get the user
            existing_user = self.get_user(identifier=identifier)
            if existing_user:
                return existing_user

            # If user doesn't exist, create new one
            return self.create_user(identifier=identifier, metadata=metadata)

        except Exception as e:
            self.logger.error(f"Error in get_or_create_user for {identifier}: {str(e)}")
            raise


