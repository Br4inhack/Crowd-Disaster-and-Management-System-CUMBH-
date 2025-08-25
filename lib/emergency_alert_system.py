"""
Emergency Alert System using Twilio SMS API
Sends automated alerts to emergency personnel (Fire Department, Police, Medical)
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import time
import threading

# Try to import Twilio, but handle gracefully if not installed
try:
    from twilio.rest import Client
    from twilio.base.exceptions import TwilioException
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    Client = None
    TwilioException = Exception
    logger.warning("Twilio not installed. SMS alerts will be disabled. Install with: pip install twilio")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class DepartmentType(Enum):
    """Emergency department types"""
    FIRE = "fire_department"
    POLICE = "police_department"
    MEDICAL = "medical_emergency"
    SECURITY = "security"
    MANAGEMENT = "management"

@dataclass
class EmergencyContact:
    """Emergency contact information"""
    name: str
    phone_number: str
    department: DepartmentType
    priority: int  # 1 = highest priority
    active: bool = True

@dataclass
class AlertConfig:
    """Alert configuration settings"""
    enabled: bool = True
    retry_attempts: int = 3
    retry_delay_minutes: int = 2
    escalation_delay_minutes: int = 5
    auto_escalate: bool = True
    include_location: bool = True
    include_timestamp: bool = True

class EmergencyAlertSystem:
    """Main emergency alert system class"""
    
    def __init__(self, config_file: str = "emergency_config.json"):
        self.config_file = config_file
        self.contacts: List[EmergencyContact] = []
        self.alert_config = AlertConfig()
        self.twilio_client: Optional[Client] = None
        self.active_alerts: Dict[str, Dict] = {}
        self.alert_history: List[Dict] = []
        
        # Load configuration
        self.load_config()
        self.setup_twilio_client()
    
    def setup_twilio_client(self):
        """Initialize Twilio client with credentials"""
        if not TWILIO_AVAILABLE:
            logger.warning("Twilio not available. SMS alerts disabled.")
            return False
            
        try:
            account_sid = os.getenv('TWILIO_ACCOUNT_SID')
            auth_token = os.getenv('TWILIO_AUTH_TOKEN')
            self.twilio_number = os.getenv('TWILIO_NUMBER')
            
            if not all([account_sid, auth_token, self.twilio_number]):
                logger.error("Missing Twilio credentials in environment variables")
                return False
                
            self.twilio_client = Client(account_sid, auth_token)
            logger.info("Twilio client initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Twilio client: {e}")
            return False
    
    def load_config(self):
        """Load emergency contacts and configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Load contacts
                self.contacts = []
                for contact_data in config_data.get('contacts', []):
                    contact = EmergencyContact(
                        name=contact_data['name'],
                        phone_number=contact_data['phone_number'],
                        department=DepartmentType(contact_data['department']),
                        priority=contact_data.get('priority', 1),
                        active=contact_data.get('active', True)
                    )
                    self.contacts.append(contact)
                
                # Load alert config
                alert_config_data = config_data.get('alert_config', {})
                self.alert_config = AlertConfig(
                    enabled=alert_config_data.get('enabled', True),
                    retry_attempts=alert_config_data.get('retry_attempts', 3),
                    retry_delay_minutes=alert_config_data.get('retry_delay_minutes', 2),
                    escalation_delay_minutes=alert_config_data.get('escalation_delay_minutes', 5),
                    auto_escalate=alert_config_data.get('auto_escalate', True),
                    include_location=alert_config_data.get('include_location', True),
                    include_timestamp=alert_config_data.get('include_timestamp', True)
                )
                
                logger.info(f"Loaded {len(self.contacts)} emergency contacts")
            else:
                # Create default configuration
                self.create_default_config()
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self.create_default_config()
    
    def create_default_config(self):
        """Create default emergency configuration"""
        default_contacts = [
            {
                "name": "Fire Department Emergency",
                "phone_number": "+1234567890",  # Replace with actual numbers
                "department": "fire_department",
                "priority": 1,
                "active": True
            },
            {
                "name": "Police Department Emergency", 
                "phone_number": "+1234567891",
                "department": "police_department",
                "priority": 1,
                "active": True
            },
            {
                "name": "Medical Emergency Services",
                "phone_number": "+1234567892",
                "department": "medical_emergency",
                "priority": 1,
                "active": True
            },
            {
                "name": "Security Team",
                "phone_number": "+1234567893",
                "department": "security",
                "priority": 2,
                "active": True
            },
            {
                "name": "Facility Management",
                "phone_number": "+1234567894",
                "department": "management",
                "priority": 3,
                "active": True
            }
        ]
        
        default_config = {
            "contacts": default_contacts,
            "alert_config": {
                "enabled": True,
                "retry_attempts": 3,
                "retry_delay_minutes": 2,
                "escalation_delay_minutes": 5,
                "auto_escalate": True,
                "include_location": True,
                "include_timestamp": True
            }
        }
        
        # Save default config
        with open(self.config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        logger.info("Created default emergency configuration")
        self.load_config()  # Reload the default config
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            config_data = {
                "contacts": [
                    {
                        "name": contact.name,
                        "phone_number": contact.phone_number,
                        "department": contact.department.value,
                        "priority": contact.priority,
                        "active": contact.active
                    }
                    for contact in self.contacts
                ],
                "alert_config": {
                    "enabled": self.alert_config.enabled,
                    "retry_attempts": self.alert_config.retry_attempts,
                    "retry_delay_minutes": self.alert_config.retry_delay_minutes,
                    "escalation_delay_minutes": self.alert_config.escalation_delay_minutes,
                    "auto_escalate": self.alert_config.auto_escalate,
                    "include_location": self.alert_config.include_location,
                    "include_timestamp": self.alert_config.include_timestamp
                }
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info("Configuration saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def create_alert_message(self, alert_level: AlertLevel, message: str, 
                           location: str = None, additional_info: Dict = None) -> str:
        """Create formatted alert message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Alert level indicators
        level_indicators = {
            AlertLevel.LOW: "🟡 LOW PRIORITY",
            AlertLevel.MEDIUM: "🟠 MEDIUM PRIORITY", 
            AlertLevel.HIGH: "🔴 HIGH PRIORITY",
            AlertLevel.CRITICAL: "🚨 CRITICAL EMERGENCY"
        }
        
        alert_msg = f"{level_indicators[alert_level]} ALERT\n\n"
        
        if self.alert_config.include_timestamp:
            alert_msg += f"⏰ Time: {timestamp}\n"
        
        if location and self.alert_config.include_location:
            alert_msg += f"📍 Location: {location}\n"
        
        alert_msg += f"📋 Details: {message}\n"
        
        if additional_info:
            alert_msg += "\n📊 Additional Information:\n"
            for key, value in additional_info.items():
                alert_msg += f"• {key}: {value}\n"
        
        alert_msg += "\n⚠️ This is an automated emergency alert from the Crowd Monitoring System."
        alert_msg += "\n🔄 Reply 'RECEIVED' to acknowledge this alert."
        
        return alert_msg
    
    def get_contacts_by_department(self, departments: List[DepartmentType]) -> List[EmergencyContact]:
        """Get active contacts filtered by department"""
        return [
            contact for contact in self.contacts 
            if contact.active and contact.department in departments
        ]
    
    def get_contacts_by_priority(self, max_priority: int = 3) -> List[EmergencyContact]:
        """Get active contacts filtered by priority level"""
        return [
            contact for contact in self.contacts 
            if contact.active and contact.priority <= max_priority
        ]
    
    def send_sms(self, phone_number: str, message: str) -> bool:
        """Send SMS message via Twilio"""
        if not TWILIO_AVAILABLE:
            logger.warning(f"Twilio not available. Would send SMS to {phone_number}: {message[:50]}...")
            return False
            
        if not self.twilio_client:
            logger.error("Twilio client not initialized")
            return False
        
        try:
            message_obj = self.twilio_client.messages.create(
                body=message,
                from_=self.twilio_number,
                to=phone_number
            )
            
            logger.info(f"SMS sent successfully to {phone_number}, SID: {message_obj.sid}")
            return True
            
        except TwilioException as e:
            logger.error(f"Twilio error sending SMS to {phone_number}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending SMS to {phone_number}: {e}")
            return False
    
    def send_emergency_alert(self, alert_level: AlertLevel, message: str,
                           departments: List[DepartmentType] = None,
                           location: str = None, additional_info: Dict = None,
                           alert_id: str = None) -> Dict:
        """Send emergency alert to specified departments"""
        
        if not self.alert_config.enabled:
            logger.warning("Alert system is disabled")
            return {"success": False, "reason": "Alert system disabled"}
        
        if not TWILIO_AVAILABLE:
            logger.warning("Twilio not available - simulating alert")
            # Continue with simulation for testing purposes
        elif not self.twilio_client:
            logger.error("Twilio client not available")
            return {"success": False, "reason": "Twilio client not available"}
        
        # Generate alert ID if not provided
        if not alert_id:
            alert_id = f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Determine target departments based on alert level
        if not departments:
            if alert_level in [AlertLevel.CRITICAL, AlertLevel.HIGH]:
                departments = [DepartmentType.FIRE, DepartmentType.POLICE, DepartmentType.MEDICAL]
            elif alert_level == AlertLevel.MEDIUM:
                departments = [DepartmentType.SECURITY, DepartmentType.MANAGEMENT]
            else:
                departments = [DepartmentType.SECURITY]
        
        # Get target contacts
        target_contacts = self.get_contacts_by_department(departments)
        
        if not target_contacts:
            logger.warning("No active contacts found for specified departments")
            return {"success": False, "reason": "No active contacts available"}
        
        # Create alert message
        alert_message = self.create_alert_message(
            alert_level, message, location, additional_info
        )
        
        # Send alerts
        results = {
            "alert_id": alert_id,
            "success": True,
            "sent_count": 0,
            "failed_count": 0,
            "contacts_notified": [],
            "failed_contacts": [],
            "timestamp": datetime.now().isoformat()
        }
        
        for contact in target_contacts:
            success = self.send_sms(contact.phone_number, alert_message)
            
            if success:
                results["sent_count"] += 1
                results["contacts_notified"].append({
                    "name": contact.name,
                    "department": contact.department.value,
                    "phone": contact.phone_number
                })
            else:
                results["failed_count"] += 1
                results["failed_contacts"].append({
                    "name": contact.name,
                    "department": contact.department.value,
                    "phone": contact.phone_number
                })
        
        # Store alert in active alerts for tracking
        self.active_alerts[alert_id] = {
            "level": alert_level.value,
            "message": message,
            "location": location,
            "departments": [dept.value for dept in departments],
            "timestamp": datetime.now(),
            "results": results,
            "acknowledged": False
        }
        
        # Add to alert history
        self.alert_history.append(results.copy())
        
        logger.info(f"Emergency alert {alert_id} sent: {results['sent_count']} successful, {results['failed_count']} failed")
        
        return results
    
    def schedule_escalation_alert(self, original_alert_id: str, delay_minutes: int = None):
        """Schedule an escalation alert if no acknowledgment received"""
        if not delay_minutes:
            delay_minutes = self.alert_config.escalation_delay_minutes
        
        def escalate():
            time.sleep(delay_minutes * 60)  # Convert to seconds
            
            if original_alert_id in self.active_alerts:
                alert_info = self.active_alerts[original_alert_id]
                
                if not alert_info["acknowledged"]:
                    # Send escalation alert
                    escalation_message = f"ESCALATION: No acknowledgment received for alert {original_alert_id}. Original message: {alert_info['message']}"
                    
                    self.send_emergency_alert(
                        AlertLevel.CRITICAL,
                        escalation_message,
                        [DepartmentType.FIRE, DepartmentType.POLICE, DepartmentType.MEDICAL],
                        alert_info["location"],
                        {"Original Alert ID": original_alert_id, "Escalation Reason": "No acknowledgment received"}
                    )
        
        if self.alert_config.auto_escalate:
            escalation_thread = threading.Thread(target=escalate)
            escalation_thread.daemon = True
            escalation_thread.start()
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "Unknown"):
        """Mark an alert as acknowledged"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id]["acknowledged"] = True
            self.active_alerts[alert_id]["acknowledged_by"] = acknowledged_by
            self.active_alerts[alert_id]["acknowledged_at"] = datetime.now()
            
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True
        
        return False
    
    def get_alert_status(self, alert_id: str) -> Dict:
        """Get status of a specific alert"""
        if alert_id in self.active_alerts:
            return self.active_alerts[alert_id]
        return None
    
    def get_active_alerts(self) -> Dict:
        """Get all active (unacknowledged) alerts"""
        return {
            alert_id: alert_info 
            for alert_id, alert_info in self.active_alerts.items()
            if not alert_info["acknowledged"]
        }
    
    def get_alert_history(self, limit: int = 50) -> List[Dict]:
        """Get recent alert history"""
        return self.alert_history[-limit:] if self.alert_history else []
    
    def test_system(self) -> Dict:
        """Test the alert system with a test message"""
        test_message = "This is a test of the emergency alert system. Please ignore this message."
        
        return self.send_emergency_alert(
            AlertLevel.LOW,
            test_message,
            [DepartmentType.SECURITY],
            "Test Location",
            {"Test": "System functionality check"}
        )
    
    def add_contact(self, contact: EmergencyContact) -> bool:
        """Add new emergency contact"""
        try:
            self.contacts.append(contact)
            self.save_config()
            logger.info(f"Added new contact: {contact.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add contact: {e}")
            return False
    
    def remove_contact(self, phone_number: str) -> bool:
        """Remove emergency contact by phone number"""
        try:
            self.contacts = [c for c in self.contacts if c.phone_number != phone_number]
            self.save_config()
            logger.info(f"Removed contact with phone: {phone_number}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove contact: {e}")
            return False
    
    def update_contact(self, phone_number: str, **kwargs) -> bool:
        """Update emergency contact information"""
        try:
            for contact in self.contacts:
                if contact.phone_number == phone_number:
                    for key, value in kwargs.items():
                        if hasattr(contact, key):
                            setattr(contact, key, value)
                    
                    self.save_config()
                    logger.info(f"Updated contact: {phone_number}")
                    return True
            
            logger.warning(f"Contact not found: {phone_number}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to update contact: {e}")
            return False
