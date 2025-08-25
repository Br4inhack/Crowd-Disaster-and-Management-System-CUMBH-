import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.emergency_alert_system import (
    EmergencyAlertSystem, EmergencyContact, AlertLevel, 
    DepartmentType, AlertConfig
)

# Page configuration
st.set_page_config(
    page_title="Emergency Alert System",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Emergency Alert System Configuration")
st.markdown("**Configure emergency contacts and alert settings for automated SMS notifications**")

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
if st.sidebar.button("← Back to Dashboard"):
    st.switch_page("streamlit_app.py")

# Initialize alert system
@st.cache_resource
def get_alert_system():
    return EmergencyAlertSystem()

alert_system = get_alert_system()

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📞 Emergency Contacts", 
    "⚙️ Alert Settings", 
    "🧪 Test System", 
    "📊 Alert History",
    "🔧 Twilio Setup"
])

with tab1:
    st.subheader("Emergency Contacts Management")
    
    # Display current contacts
    if alert_system.contacts:
        st.write("**Current Emergency Contacts:**")
        
        contacts_data = []
        for contact in alert_system.contacts:
            contacts_data.append({
                "Name": contact.name,
                "Phone": contact.phone_number,
                "Department": contact.department.value.replace('_', ' ').title(),
                "Priority": contact.priority,
                "Active": "✅" if contact.active else "❌"
            })
        
        contacts_df = pd.DataFrame(contacts_data)
        st.dataframe(contacts_df, use_container_width=True)
        
        # Contact management
        st.write("**Manage Contacts:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Edit Contact Status:**")
            contact_to_edit = st.selectbox(
                "Select contact to edit",
                options=[f"{c.name} ({c.phone_number})" for c in alert_system.contacts],
                key="edit_contact"
            )
            
            if contact_to_edit:
                selected_contact = None
                for contact in alert_system.contacts:
                    if f"{contact.name} ({contact.phone_number})" == contact_to_edit:
                        selected_contact = contact
                        break
                
                if selected_contact:
                    new_active_status = st.checkbox(
                        "Active", 
                        value=selected_contact.active,
                        key="contact_active"
                    )
                    
                    new_priority = st.number_input(
                        "Priority (1=highest)", 
                        min_value=1, 
                        max_value=5, 
                        value=selected_contact.priority,
                        key="contact_priority"
                    )
                    
                    if st.button("Update Contact", key="update_btn"):
                        success = alert_system.update_contact(
                            selected_contact.phone_number,
                            active=new_active_status,
                            priority=new_priority
                        )
                        if success:
                            st.success("Contact updated successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to update contact")
        
        with col2:
            st.write("**Remove Contact:**")
            contact_to_remove = st.selectbox(
                "Select contact to remove",
                options=[f"{c.name} ({c.phone_number})" for c in alert_system.contacts],
                key="remove_contact"
            )
            
            if st.button("Remove Contact", type="secondary", key="remove_btn"):
                if contact_to_remove:
                    phone = contact_to_remove.split("(")[1].split(")")[0]
                    success = alert_system.remove_contact(phone)
                    if success:
                        st.success("Contact removed successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to remove contact")
    
    else:
        st.info("No emergency contacts configured. Add contacts below.")
    
    # Add new contact
    st.markdown("---")
    st.subheader("Add New Emergency Contact")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        new_name = st.text_input("Contact Name", placeholder="Fire Department Emergency")
        new_phone = st.text_input("Phone Number", placeholder="+1234567890")
    
    with col2:
        new_department = st.selectbox(
            "Department",
            options=[dept.value.replace('_', ' ').title() for dept in DepartmentType],
            key="new_dept"
        )
        new_priority = st.number_input("Priority (1=highest)", min_value=1, max_value=5, value=1)
    
    with col3:
        new_active = st.checkbox("Active", value=True, key="new_active")
        
        if st.button("Add Contact", type="primary"):
            if new_name and new_phone:
                # Convert department back to enum
                dept_enum = None
                for dept in DepartmentType:
                    if dept.value.replace('_', ' ').title() == new_department:
                        dept_enum = dept
                        break
                
                new_contact = EmergencyContact(
                    name=new_name,
                    phone_number=new_phone,
                    department=dept_enum,
                    priority=new_priority,
                    active=new_active
                )
                
                success = alert_system.add_contact(new_contact)
                if success:
                    st.success("Contact added successfully!")
                    st.rerun()
                else:
                    st.error("Failed to add contact")
            else:
                st.error("Please fill in all required fields")

with tab2:
    st.subheader("Alert System Configuration")
    
    # Current settings
    config = alert_system.alert_config
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Basic Settings:**")
        
        new_enabled = st.checkbox("Enable Alert System", value=config.enabled)
        new_retry_attempts = st.number_input(
            "Retry Attempts", 
            min_value=1, 
            max_value=10, 
            value=config.retry_attempts
        )
        new_retry_delay = st.number_input(
            "Retry Delay (minutes)", 
            min_value=1, 
            max_value=30, 
            value=config.retry_delay_minutes
        )
    
    with col2:
        st.write("**Advanced Settings:**")
        
        new_auto_escalate = st.checkbox("Auto Escalate Alerts", value=config.auto_escalate)
        new_escalation_delay = st.number_input(
            "Escalation Delay (minutes)", 
            min_value=1, 
            max_value=60, 
            value=config.escalation_delay_minutes
        )
        new_include_location = st.checkbox("Include Location in Alerts", value=config.include_location)
        new_include_timestamp = st.checkbox("Include Timestamp in Alerts", value=config.include_timestamp)
    
    if st.button("Save Alert Configuration", type="primary"):
        alert_system.alert_config = AlertConfig(
            enabled=new_enabled,
            retry_attempts=new_retry_attempts,
            retry_delay_minutes=new_retry_delay,
            escalation_delay_minutes=new_escalation_delay,
            auto_escalate=new_auto_escalate,
            include_location=new_include_location,
            include_timestamp=new_include_timestamp
        )
        
        success = alert_system.save_config()
        if success:
            st.success("Configuration saved successfully!")
        else:
            st.error("Failed to save configuration")
    
    # Alert level configuration
    st.markdown("---")
    st.subheader("Alert Level Guidelines")
    
    alert_info = {
        "🟡 LOW": "Security team only - Minor issues, non-urgent",
        "🟠 MEDIUM": "Security + Management - Moderate issues requiring attention", 
        "🔴 HIGH": "Fire + Police + Medical - Serious safety concerns",
        "🚨 CRITICAL": "All departments - Immediate life-threatening emergency"
    }
    
    for level, description in alert_info.items():
        st.write(f"**{level}**: {description}")

with tab3:
    st.subheader("Test Emergency Alert System")
    
    st.warning("⚠️ This will send actual SMS messages to configured contacts!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_level = st.selectbox(
            "Test Alert Level",
            options=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            index=0
        )
        
        test_message = st.text_area(
            "Test Message",
            value="This is a test of the emergency alert system. Please ignore this message.",
            height=100
        )
        
        test_location = st.text_input("Test Location", value="Test Facility - Main Building")
    
    with col2:
        test_departments = st.multiselect(
            "Target Departments",
            options=[dept.value.replace('_', ' ').title() for dept in DepartmentType],
            default=["Security"]
        )
        
        include_additional_info = st.checkbox("Include Additional Test Info")
        
        if include_additional_info:
            additional_info = {
                "Test ID": f"TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "System": "Crowd Monitoring Alert System",
                "Operator": "System Administrator"
            }
        else:
            additional_info = None
    
    if st.button("🧪 Send Test Alert", type="secondary"):
        if test_message and test_departments:
            # Convert departments back to enums
            dept_enums = []
            for dept_str in test_departments:
                for dept in DepartmentType:
                    if dept.value.replace('_', ' ').title() == dept_str:
                        dept_enums.append(dept)
                        break
            
            # Send test alert
            result = alert_system.send_emergency_alert(
                AlertLevel(test_level.lower()),
                test_message,
                dept_enums,
                test_location,
                additional_info
            )
            
            if result["success"]:
                st.success(f"✅ Test alert sent successfully!")
                st.write(f"**Alert ID:** {result['alert_id']}")
                st.write(f"**Messages Sent:** {result['sent_count']}")
                st.write(f"**Failed:** {result['failed_count']}")
                
                if result["contacts_notified"]:
                    st.write("**Contacts Notified:**")
                    for contact in result["contacts_notified"]:
                        st.write(f"• {contact['name']} ({contact['department']})")
                
                if result["failed_contacts"]:
                    st.error("**Failed Contacts:**")
                    for contact in result["failed_contacts"]:
                        st.write(f"• {contact['name']} ({contact['department']})")
            else:
                st.error(f"❌ Test alert failed: {result.get('reason', 'Unknown error')}")
        else:
            st.error("Please fill in all required fields")
    
    # System status
    st.markdown("---")
    st.subheader("System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        active_contacts = len([c for c in alert_system.contacts if c.active])
        st.metric("Active Contacts", active_contacts)
    
    with col2:
        twilio_status = "✅ Connected" if alert_system.twilio_client else "❌ Not Connected"
        st.metric("Twilio Status", twilio_status)
    
    with col3:
        system_status = "✅ Enabled" if alert_system.alert_config.enabled else "❌ Disabled"
        st.metric("Alert System", system_status)

with tab4:
    st.subheader("Alert History")
    
    history = alert_system.get_alert_history()
    active_alerts = alert_system.get_active_alerts()
    
    # Active alerts
    if active_alerts:
        st.write("**🔴 Active Alerts (Unacknowledged):**")
        for alert_id, alert_info in active_alerts.items():
            with st.expander(f"Alert {alert_id} - {alert_info['level'].upper()}", expanded=True):
                st.write(f"**Message:** {alert_info['message']}")
                st.write(f"**Time:** {alert_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**Location:** {alert_info.get('location', 'Not specified')}")
                st.write(f"**Departments:** {', '.join(alert_info['departments'])}")
                
                if st.button(f"Mark as Acknowledged", key=f"ack_{alert_id}"):
                    alert_system.acknowledge_alert(alert_id, "Manual Acknowledgment")
                    st.success("Alert acknowledged!")
                    st.rerun()
    else:
        st.success("✅ No active alerts")
    
    # Alert history
    if history:
        st.markdown("---")
        st.write("**📊 Recent Alert History:**")
        
        history_data = []
        for alert in history[-20:]:  # Last 20 alerts
            history_data.append({
                "Alert ID": alert["alert_id"],
                "Timestamp": alert["timestamp"],
                "Sent": alert["sent_count"],
                "Failed": alert["failed_count"],
                "Success Rate": f"{(alert['sent_count'] / (alert['sent_count'] + alert['failed_count']) * 100):.1f}%" if (alert['sent_count'] + alert['failed_count']) > 0 else "0%"
            })
        
        if history_data:
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)
        
        # Summary statistics
        total_sent = sum(alert["sent_count"] for alert in history)
        total_failed = sum(alert["failed_count"] for alert in history)
        success_rate = (total_sent / (total_sent + total_failed) * 100) if (total_sent + total_failed) > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Alerts Sent", len(history))
        with col2:
            st.metric("Messages Delivered", total_sent)
        with col3:
            st.metric("Success Rate", f"{success_rate:.1f}%")
    else:
        st.info("No alert history available")

with tab5:
    st.subheader("Twilio API Configuration")
    
    st.write("**Setup Instructions:**")
    st.markdown("""
    1. **Create a Twilio Account:**
       - Go to [twilio.com](https://twilio.com) and sign up
       - Verify your account and get a phone number
    
    2. **Get Your Credentials:**
       - Find your Account SID and Auth Token in the Twilio Console
       - Note your Twilio phone number
    
    3. **Set Environment Variables:**
       - `TWILIO_ACCOUNT_SID`: Your account SID
       - `TWILIO_AUTH_TOKEN`: Your auth token  
       - `TWILIO_NUMBER`: Your Twilio phone number (in E.164 format, e.g., +1234567890)
    """)
    
    # Check current environment variables
    st.write("**Current Environment Status:**")
    
    env_vars = {
        "TWILIO_ACCOUNT_SID": os.getenv('TWILIO_ACCOUNT_SID'),
        "TWILIO_AUTH_TOKEN": os.getenv('TWILIO_AUTH_TOKEN'),
        "TWILIO_NUMBER": os.getenv('TWILIO_NUMBER')
    }
    
    for var_name, var_value in env_vars.items():
        if var_value:
            if var_name == "TWILIO_AUTH_TOKEN":
                display_value = f"{'*' * (len(var_value) - 4)}{var_value[-4:]}" if len(var_value) > 4 else "****"
            else:
                display_value = var_value
            st.success(f"✅ {var_name}: {display_value}")
        else:
            st.error(f"❌ {var_name}: Not set")
    
    # Environment setup helper
    st.markdown("---")
    st.write("**Environment Setup Helper:**")
    
    with st.expander("💡 How to set environment variables"):
        st.markdown("""
        **Windows (Command Prompt):**
        ```
        set TWILIO_ACCOUNT_SID=your_account_sid_here
        set TWILIO_AUTH_TOKEN=your_auth_token_here
        set TWILIO_NUMBER=+1234567890
        ```
        
        **Windows (PowerShell):**
        ```
        $env:TWILIO_ACCOUNT_SID="your_account_sid_here"
        $env:TWILIO_AUTH_TOKEN="your_auth_token_here"
        $env:TWILIO_NUMBER="+1234567890"
        ```
        
        **Linux/Mac:**
        ```
        export TWILIO_ACCOUNT_SID=your_account_sid_here
        export TWILIO_AUTH_TOKEN=your_auth_token_here
        export TWILIO_NUMBER=+1234567890
        ```
        
        **For permanent setup, add these to your system environment variables or .env file**
        """)
    
    # Test connection
    if st.button("🔗 Test Twilio Connection"):
        if alert_system.twilio_client:
            try:
                # Try to get account info
                account = alert_system.twilio_client.api.accounts(alert_system.twilio_client.account_sid).fetch()
                st.success(f"✅ Connected to Twilio! Account: {account.friendly_name}")
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")
        else:
            st.error("❌ Twilio client not initialized. Check your environment variables.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🚨 Emergency Alert System | Powered by Twilio SMS API</p>
        <p>⚠️ Ensure all emergency contacts are verified and up-to-date</p>
    </div>
    """,
    unsafe_allow_html=True
)
