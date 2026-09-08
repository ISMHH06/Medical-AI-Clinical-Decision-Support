import os
import redivis

# Optional: set token directly in Python if preferred
# os.environ["REDIVIS_API_TOKEN"] = "YOUR_COPIED_TOKEN_HERE"

user = redivis.user("ISMAILHH")

for wf_name, wf_id in [("ISMHH2", "s054"), ("ISMHH", "hkta")]:
    print(f"\n==========================================")
    print(f"Inspecting Workflow: {wf_name} ({wf_id})")
    print(f"==========================================")
    try:
        workflow = user.workflow(f"{wf_name}:{wf_id}")
        tables = workflow.list_tables()
        print(f"Found {len(tables)} tables:")
        for t in tables:
            t.get()
            print(f"  └── Table Name        : {t.name}")
            print(f"      Qualified Reference: {t.qualified_reference}")
    except Exception as e:
        print(f"Error inspecting {wf_name}: {e}")