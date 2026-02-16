# Inventory Management System

We are building a modern inventory management system for warehouse operations.
The system needs to handle tracking of physical goods across multiple locations.

The platform must support barcode scanning and RFID tag reading for item identification.
It should integrate with the existing ERP system via REST APIs.
The system shall provide real-time stock level visibility across all warehouses.

Users need to be able to perform stock counts, transfers between locations, and
generate reports on inventory levels and movement patterns.

The application must ensure data consistency even during network partitions between
warehouse nodes. It should support offline operation with automatic synchronization
when connectivity is restored.

Performance is critical: the system must handle at least 500 scan events per minute
per warehouse without degradation. The dashboard should load within 2 seconds.

Security requirements include role-based access control, audit logging of all
inventory changes, and encrypted communication between all components.
